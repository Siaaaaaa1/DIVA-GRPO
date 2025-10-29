# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agnostic initialization with HuggingFace models.
"""

import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, Dict, List, Optional, Type

import numpy as np
import ray
import torch
from ray.experimental.tqdm_ray import tqdm
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from ..single_controller.base import Worker
from ..single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from ..single_controller.ray.base import create_colocated_worker_cls
from ..utils import torch_functional as VF
from ..utils.checkpoint import CHECKPOINT_TRACKER, remove_obsolete_ckpt
from ..utils.logger import Tracker
from ..utils.py_functional import convert_dict_to_str, timer
from ..utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import FunctionRewardManager
from . import core_algos
from .config import PPOConfig
from .core_algos import AdvantageEstimator, FixedKLController, KLController, compute_kl, get_kl_controller
from .metrics import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics, reduce_metrics
from ..difficulty_variation.difficulty_utils import rotate_image, add_gaussian_noise, add_salt_pepper_noise
import datetime
import fcntl
from datasets import Dataset, DatasetDict


class Role(IntEnum):
    """Roles used to map worker processes."""
    Actor = auto()
    Rollout = auto()
    ActorRollout = auto()
    Critic = auto()
    RefPolicy = auto()
    RewardModel = auto()
    ActorRolloutRef = auto()


@dataclass
class ResourcePoolManager:
    """Defines a resource pool specification. Resource pool will be initialized first."""
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool
        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        return self.resource_pool_dict[self.mapping[role]]

    def get_num_gpus(self) -> int:
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        gpus_available = ray.available_resources().get("GPU", 0)
        gpus_required = self.get_num_gpus()
        if gpus_available < gpus_required:
            raise ValueError(f"Total available GPUs {gpus_available} is less than total desired GPUs {gpus_required}.")


def apply_kl_penalty(data: DataProto, kl_ctrl: KLController, kl_penalty="kl"):
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    response_mask = data.batch["response_mask"]

    kld = compute_kl(data.batch["old_log_probs"], data.batch["ref_log_probs"], kl_penalty=kl_penalty)
    kld = kld * response_mask

    data.batch["token_level_rewards"] = token_level_scores - kl_ctrl.kl_coef * kld
    current_kl = VF.masked_mean(kld, mask=response_mask, dim=-1)
    current_kl = torch.mean(current_kl, dim=0).item()
    metrics = {"critic/kl": current_kl, "critic/kl_coef": kl_ctrl.kl_coef}
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    return data, metrics


import math
import torch


def multiplier(difficult, advantage):
    k = 0.1
    multiplier = torch.ones_like(advantage)
    nonzero_mask = (advantage != 0)
    sign = torch.where(advantage > 0, 1.0, -1.0)
    multiplier[nonzero_mask] = torch.exp(k * difficult * sign[nonzero_mask])
    return multiplier


def weighted_advantage(difficult, advantage):
    return advantage * multiplier(difficult, advantage)


def adjust_low_reward_advantages(
    global_advantages: torch.Tensor,
    global_index: list,
    token_level_rewards: torch.Tensor,
    threshold: float = 0.15,
    scale_factor: float = 0.1
) -> torch.Tensor:
    """Adjusts advantages corresponding to low-reward global IDs."""
    global_id_to_rewards = {}
    for gid, rewards in zip(global_index, token_level_rewards):
        if gid not in global_id_to_rewards:
            global_id_to_rewards[gid] = []
        global_id_to_rewards[gid].append(rewards.sum().item())

    low_reward_global_ids = {
        gid for gid, rewards in global_id_to_rewards.items()
        if all(r < threshold for r in rewards)
    }

    adjusted_advantages = global_advantages.clone()
    for i, gid in enumerate(global_index):
        if gid in low_reward_global_ids:
            adjusted_advantages[i] = adjusted_advantages[i] * scale_factor

    return adjusted_advantages


def log_difficulty_update(self, id_, old_diff, new_diff, math_reward):
    log_entry = {
        "id": id_,
        "old_difficulty": old_diff,
        "new_difficulty": new_diff,
        "math_reward": math_reward,
        "timestamp": datetime.datetime.now().isoformat()
    }
    try:
        with open("difficulty_updates.log", "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        print(f"Error logging difficulty update: {e}")


class RayPPOTrainer:
    """
    Note: this trainer runs on the driver process on a single CPU/GPU node.
    """

    def __init__(
        self,
        config: PPOConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        train_dataloader: StatefulDataLoader,
        val_dataloader: StatefulDataLoader,
        role_worker_mapping: dict[Role, Type[Worker]],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: Type[RayWorkerGroup] = RayWorkerGroup,
        reward_fn: Optional[FunctionRewardManager] = None,
        val_reward_fn: Optional[FunctionRewardManager] = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.val_reward_score = 0.0
        self.best_val_reward_score = -1.0
        self.best_global_step = None

        self.hybrid_engine = config.worker.hybrid_engine
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reward_model = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        if config.algorithm.disable_kl:
            self.use_reference_policy = False
            self.kl_ctrl = FixedKLController(init_kl_coef=0.0)
            print("KL is disabled, no KL metrics will be logged. Please set `kl_coef=0` to log KL metrics.")
        else:
            self.use_reference_policy = True
            self.kl_ctrl = get_kl_controller(config.algorithm)

        if config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        else:
            self.use_critic = False

        if config.algorithm.adv_estimator not in list(AdvantageEstimator):
            raise NotImplementedError(f"Unknown advantage estimator: {config.algorithm.adv_estimator}.")

        if config.data.rollout_batch_size % config.worker.actor.global_batch_size != 0:
            raise ValueError("Rollout batch size must be divisible by actor global batch size.")

        if (
            config.data.rollout_batch_size * config.worker.rollout.n
        ) % config.worker.actor.micro_batch_size_per_device_for_experience != 0:
            raise ValueError(
                "Rollout batch size * rollout.n must be divisible by actor micro batch size for experience."
            )

        if self.use_critic:
            if config.data.rollout_batch_size % config.worker.critic.global_batch_size != 0:
                raise ValueError("Rollout batch size must be divisible by critic global batch size.")

            if (
                config.data.rollout_batch_size * config.worker.rollout.n
            ) % config.worker.critic.micro_batch_size_per_device_for_experience != 0:
                raise ValueError(
                    "Rollout batch size * rollout.n must be divisible by critic micro batch size for experience."
                )

        if (config.algorithm.adv_estimator in (AdvantageEstimator.GRPO, AdvantageEstimator.RLOO)
            and config.worker.rollout.n == 1):
            raise ValueError("GRPO and RLOO algorithm need `config.worker.rollout.n > 1`.")

        if config.trainer.max_steps is not None:
            self.training_steps = config.trainer.max_steps
        elif config.trainer.Difficulty_Adaptation is True:
            self.training_steps = len(train_dataloader) * config.trainer.total_epochs
        elif config.data.mini_rollout_batch_size is not None:
            num_examples = len(train_dataloader) * config.data.mini_rollout_batch_size
            self.training_steps = num_examples // config.data.rollout_batch_size * config.trainer.total_epochs
        else:
            self.training_steps = len(train_dataloader) * config.trainer.total_epochs * (config.trainer.Varient_Num + 1)

        config.worker.actor.optim.training_steps = self.training_steps
        config.worker.critic.optim.training_steps = self.training_steps
        print(f"Total training steps: {self.training_steps}")

    def init_workers(self) -> None:
        """Init resource pool and worker group."""
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRolloutRef)
            actor_rollout_ref_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRolloutRef], config=self.config.worker, role="actor_rollout_ref"
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout_ref"] = actor_rollout_ref_cls
        else:
            raise NotImplementedError

        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], config=self.config.worker, role="critic"
            )
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        if self.use_reward_model:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RewardModel], config=self.config.worker, role="reward"
            )
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        all_wg: Dict[str, FSDPWorker] = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reward_model:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        self.actor_rollout_ref_wg = all_wg["actor_rollout_ref"]
        self.actor_rollout_ref_wg.init_model()

    def filter_dataset_by_difficulty(self):
        """
        Remove samples with difficulty < 1 or > 9 from the training dataset.
        Returns:
            bool: True if any samples were removed, else False.
        """
        print("\n[DEBUG] Starting removal of samples with difficulty <1 or >9...")

        before_count = len(self.train_dataloader.dataset)

        def filter_outliers(example):
            return 1 <= example['difficulty'] <= 9

        self.train_dataloader.dataset.dataset = self.train_dataloader.dataset.dataset.filter(
            filter_outliers,
            num_proc=4,
            desc="Filtering outliers"
        )

        after_count = len(self.train_dataloader.dataset)
        removed_count = before_count - after_count

        print("\n[DEBUG] Removal summary:")
        print(f"  - Original dataset size: {before_count}")
        print(f"  - New dataset size: {after_count}")
        print(f"  - Removed samples: {removed_count}")

        return removed_count > 0

    def _save_dataset(self) -> None:
        """Save the training dataset as a parquet file under the checkpoint directory (GPU 0 only)."""
        try:
            import torch
            if torch.cuda.is_available() and torch.cuda.current_device() != 0:
                print("Not on GPU 0, skipping dataset save.")
                return

            import pyarrow.parquet as pq  # noqa: F401 (import ensures engine availability)
            import pandas as pd

            dataset = self.train_dataloader.dataset

            if hasattr(dataset, 'dataset') and isinstance(dataset.dataset, (Dataset, DatasetDict)):
                hf_dataset = dataset.dataset
                if isinstance(hf_dataset, DatasetDict):
                    hf_dataset = next(iter(hf_dataset.values()))
                df = hf_dataset.to_pandas()

            elif hasattr(dataset, 'to_pandas'):
                df = dataset.to_pandas()

            elif hasattr(dataset, '__array__'):
                df = pd.DataFrame(dataset)

            elif hasattr(dataset, '__iter__'):
                data_list = list(dataset)
                if data_list and isinstance(data_list[0], dict):
                    df = pd.DataFrame(data_list)
                else:
                    df = pd.DataFrame({'data': data_list})
            else:
                raise ValueError("Unsupported dataset type - cannot convert to DataFrame.")

            folder_path = self.config.trainer.save_checkpoint_path
            os.makedirs(folder_path, exist_ok=True)
            parquet_path = os.path.join(folder_path, "MMK12_Adapter.parquet")

            df.to_parquet(parquet_path)
            print(f"Dataset saved to {parquet_path}")

        except Exception as e:
            print(f"Failed to save dataset: {str(e)}")
            raise

    def _save_checkpoint(self) -> None:
        # path: {save_checkpoint_path}/global_step_{global_step}/{actor,critic}
        if self.val_reward_score > self.best_val_reward_score:
            self.best_val_reward_score = self.val_reward_score
            self.best_global_step = self.global_step

        remove_obsolete_ckpt(
            self.config.trainer.save_checkpoint_path,
            self.global_step,
            self.best_global_step,
            self.config.trainer.save_limit,
        )
        folder_path = os.path.join(self.config.trainer.save_checkpoint_path, f"global_step_{self.global_step}")
        actor_path = os.path.join(folder_path, "actor")
        self.actor_rollout_ref_wg.save_checkpoint(actor_path, save_model_only=self.config.trainer.save_model_only)

        if self.use_critic:
            critic_path = os.path.join(folder_path, "critic")
            self.critic_wg.save_checkpoint(critic_path, save_model_only=self.config.trainer.save_model_only)

        dataloader_path = os.path.join(folder_path, "dataloader.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_path)

        # Optional dataset operations:
        # self._save_dataset()
        self.filter_dataset_by_difficulty()

        checkpointer_tracker_info = {
            "best_global_step": self.best_global_step,
            "best_val_reward_score": round(self.best_val_reward_score, 4),
            "last_global_step": self.global_step,
            "last_actor_path": os.path.abspath(actor_path),
        }
        checkpointer_tracker_path = os.path.join(self.config.trainer.save_checkpoint_path, CHECKPOINT_TRACKER)
        with open(checkpointer_tracker_path, "w") as f:
            json.dump(checkpointer_tracker_info, f, ensure_ascii=False, indent=2)

    def _load_checkpoint(self) -> None:
        if self.config.trainer.load_checkpoint_path is None:
            return

        if "global_step_" not in self.config.trainer.load_checkpoint_path.strip(os.path.sep).split(os.path.sep)[-1]:
            raise ValueError("`load_checkpoint_path` should end with `global_step_*`.")

        print(f"Load from checkpoint: {self.config.trainer.load_checkpoint_path}.")
        self.global_step = int(self.config.trainer.load_checkpoint_path.strip(os.path.sep).split("global_step_")[-1])
        actor_path = os.path.join(self.config.trainer.load_checkpoint_path, "actor")
        self.actor_rollout_ref_wg.load_checkpoint(actor_path)
        if self.use_critic:
            critic_path = os.path.join(self.config.trainer.load_checkpoint_path, "critic")
            self.critic_wg.load_checkpoint(critic_path)

        dataloader_path = os.path.join(self.config.trainer.load_checkpoint_path, "dataloader.pt")
        if os.path.exists(dataloader_path):
            dataloader_state_dict = torch.load(dataloader_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"No dataloader state found at {dataloader_path}, will start from scratch.")

    def _maybe_log_val_generations(
        self, inputs: List[str], outputs: List[str], labels: List[str], scores: List[float]
    ) -> None:
        """Log a table of validation samples."""
        if self.config.trainer.val_generations_to_log <= 0:
            return

        samples = list(zip(inputs, outputs, labels, scores))
        samples.sort(key=lambda x: x[0])

        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        samples = samples[: self.config.trainer.val_generations_to_log]
        self.logger.log_generation(samples, self.global_step)

    def _validate(self) -> Dict[str, Any]:
        reward_tensor_lst = []
        sample_inputs, sample_outputs, sample_labels, sample_scores = [], [], [], []
        reward_metrics_lst = defaultdict(list)
        print("Start validation...")
        self.actor_rollout_ref_wg.prepare_rollout_engine()
        for batch_dict in self.val_dataloader:
            test_batch = DataProto.from_single_dict(batch_dict)
            test_gen_batch = test_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
            )
            repeat_times = self.config.worker.rollout.val_override_config.get("n", 1)
            test_gen_batch.meta_info = self.config.worker.rollout.val_override_config
            test_gen_batch.meta_info["min_pixels"] = self.config.data.min_pixels
            test_gen_batch.meta_info["max_pixels"] = self.config.data.max_pixels
            test_gen_batch.meta_info["video_fps"] = self.config.data.video_fps

            test_gen_batch, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_ref_wg.world_size)
            test_output_gen_batch = self.actor_rollout_ref_wg.generate_sequences(test_gen_batch)
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch, pad_size=pad_size * repeat_times)

            test_batch = test_batch.repeat(repeat_times=repeat_times, interleave=True)
            test_batch = test_batch.union(test_output_gen_batch)

            reward_tensor, reward_metrics = ray.get(self.val_reward_fn.compute_reward.remote(test_batch))

            input_ids = test_batch.batch["prompts"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            output_ids = test_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_inputs.extend(input_texts)
            sample_outputs.extend(output_texts)
            sample_labels.extend(test_batch.non_tensor_batch["ground_truth"].tolist())
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            for key, value in reward_metrics.items():
                reward_metrics_lst[key].extend(value)

        self.actor_rollout_ref_wg.release_rollout_engine()
        self._maybe_log_val_generations(sample_inputs, sample_outputs, sample_labels, sample_scores)
        self.val_reward_score = torch.cat(reward_tensor_lst, dim=0).sum(-1).mean().item()
        val_reward_metrics = {f"val/{key}_reward": value for key, value in reduce_metrics(reward_metrics_lst).items()}
        print("Finish validation.")
        return {"val/reward_score": self.val_reward_score, **val_reward_metrics}

    def _balance_batch(self, batch: DataProto, metrics: Dict[str, Any], logging_prefix: str = "global_seqlen") -> None:
        """Reorder data so each DP rank sees similar total tokens."""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()
        world_size = self.actor_rollout_ref_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _make_batch_data(self, metrics: Dict[str, Any]) -> DataProto:
        """Create one training batch and collect optional reward metrics."""
        batch = None
        all_metrics = defaultdict(list)
        num_try_make_batch = 0
        print("Start generating batch...")

        while True:
            num_try_make_batch += 1
            try:
                batch_dict = next(self.data_iterator)
            except StopIteration:
                self.data_iterator = iter(self.train_dataloader)
                batch_dict = next(self.data_iterator)

            meta_info = {
                "min_pixels": self.config.data.min_pixels,
                "max_pixels": self.config.data.max_pixels,
                "video_fps": self.config.data.video_fps,
            }

            new_batch: DataProto = DataProto.from_single_dict(batch_dict, meta_info=meta_info)

            gen_batch = new_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                meta_info_keys=["min_pixels", "max_pixels", "video_fps"],
            )

            gen_batch_output = self.actor_rollout_ref_wg.generate_sequences(gen_batch)

            if self.config.algorithm.adv_estimator == "remax":
                gen_baseline_batch = deepcopy(gen_batch)
                gen_baseline_batch.meta_info["temperature"] = 0
                gen_baseline_batch.meta_info["n"] = 1

                gen_baseline_output = self.actor_rollout_ref_wg.generate_sequences(gen_baseline_batch)

                new_batch = new_batch.union(gen_baseline_output)
                reward_baseline_tensor, _ = ray.get(self.reward_fn.compute_reward.remote(new_batch))
                reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                new_batch.batch["reward_baselines"] = reward_baseline_tensor
                del gen_baseline_batch, gen_baseline_output

            new_batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
            )
            if self.config.trainer.Difficulty_Adaptation:
                ids = new_batch.non_tensor_batch["id"]

                id_counts = {}
                for id_val in ids:
                    id_counts[id_val] = id_counts.get(id_val, 0) + 1

                unique_ids = list(id_counts.keys())
                id_to_uid = {id_val: str(uuid.uuid4()) for id_val in unique_ids}

                global_uids = np.array([id_to_uid[id_val] for id_val in ids], dtype=object)
                new_batch.non_tensor_batch["global_uid"] = global_uids

            new_batch = new_batch.repeat(repeat_times=self.config.worker.rollout.n, interleave=True)
            new_batch = new_batch.union(gen_batch_output)

            if self.config.algorithm.online_filtering:
                reward_tensor, reward_metrics = ray.get(self.reward_fn.compute_reward.remote(new_batch))
                new_batch.batch["token_level_scores"] = reward_tensor

                for k, v in reward_metrics.items():
                    all_metrics[k].extend(v)

                filter_scores = reward_metrics[self.config.algorithm.filter_key]
                uids = new_batch.non_tensor_batch["uid"]

                uid2scores = defaultdict(list)
                for uid, score in zip(uids, filter_scores):
                    uid2scores[uid].append(score)

                uid2mean = {uid: np.mean(scores) for uid, scores in uid2scores.items()}

                kept_uids = [
                    uid
                    for uid, avg_score in uid2mean.items()
                    if avg_score > self.config.algorithm.filter_low and avg_score < self.config.algorithm.filter_high
                ]
                kept_sample_idxs = [idx for idx, uid in enumerate(uids) if uid in kept_uids]
                new_batch = new_batch[kept_sample_idxs]

            batch = DataProto.concat([batch, new_batch]) if batch is not None else new_batch

            current_batch_size = len(batch) // self.config.worker.rollout.n
            rollout_batch_size = self.config.data.rollout_batch_size

            if current_batch_size < rollout_batch_size:
                print(f"{current_batch_size=} < {rollout_batch_size=}")
                max_try_make_batch = self.config.trainer.max_try_make_batch

                if max_try_make_batch <= 0 or num_try_make_batch < max_try_make_batch:
                    print(f"{num_try_make_batch=}. Continue generating...")
                else:
                    raise ValueError(
                        f"{num_try_make_batch=} >= {max_try_make_batch=}. Generated too many. Please check your data."
                    )
            else:
                print(f"{current_batch_size=} >= {rollout_batch_size=}. Finish generating.")
                if self.config.algorithm.online_filtering:
                    metrics.update({f"reward/{k}": v for k, v in reduce_metrics(all_metrics).items()})

                return batch[: self.config.data.rollout_batch_size * self.config.worker.rollout.n]

    def _compute_difficult_adaption_advantage(
        self,
        data: DataProto,
        adv_estimator: AdvantageEstimator,
        gamma: float = 1.0,
        lam: float = 1.0
    ):
        token_level_rewards = data.batch["token_level_rewards"]
        response_mask = data.batch["response_mask"]
        index = data.non_tensor_batch["uid"]
        global_index = data.non_tensor_batch["global_uid"]
        difficult = data.non_tensor_batch["difficulty"]
        category = data.non_tensor_batch["category"]
        id = data.non_tensor_batch["id"]
        alpfa = 1
        beta = 1

        if adv_estimator == AdvantageEstimator.GRPO:
            local_advantages, local_returns = core_algos.compute_grpo_outcome_advantage(
                token_level_rewards, response_mask, index
            )
            global_advantages, global_returns = core_algos.compute_grpo_outcome_advantage(
                token_level_rewards, response_mask, global_index
            )

            advantages = alpfa * local_advantages + beta * global_advantages
            returns = alpfa * local_advantages + beta * global_advantages
            weighted_global_advantages = global_advantages

            self._save_full_vectors_to_json(
                data, local_advantages, global_advantages, weighted_global_advantages, advantages, token_level_rewards
            )

        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        return data

    def _compute_difficult_adaption_advantage_difficulty(
        self,
        data: DataProto,
        adv_estimator: AdvantageEstimator,
        gamma: float = 1.0,
        lam: float = 1.0,
        weight_mode: str = "weight_before_norm",
        Adjust_Low_Reward_Local: bool = False,
        Adjust_Low_Reward_Global: bool = False
    ):
        token_level_rewards = data.batch["token_level_rewards"]
        response_mask = data.batch["response_mask"]
        index = data.non_tensor_batch["uid"]
        global_index = data.non_tensor_batch["global_uid"]
        difficult = data.non_tensor_batch["difficulty"]
        category = data.non_tensor_batch["category"]
        id = data.non_tensor_batch["id"]
        old_log_probs = data.batch["old_log_probs"]
        alpfa = 1
        beta = 1

        # Compute mean difficulty per global_id
        global_difficult_means = {}
        for gid, diff in zip(global_index, difficult):
            if gid not in global_difficult_means:
                global_difficult_means[gid] = []
            global_difficult_means[gid].append(diff)
        for gid in global_difficult_means:
            global_difficult_means[gid] = sum(global_difficult_means[gid]) / len(global_difficult_means[gid])

        # Difference between each sample's difficulty and its group's mean
        difficult_diffs = []
        for gid, diff in zip(global_index, difficult):
            difficult_diffs.append(diff - global_difficult_means[gid])

        if adv_estimator == AdvantageEstimator.GRPO:
            def to_numpy(x):
                return x.cpu().numpy() if hasattr(x, 'cpu') else np.array(x)
            token_rewards = to_numpy(token_level_rewards)
            scores = token_rewards.sum(axis=-1)

            difficult_map = {}
            for sample_id, diff, cat in zip(id, difficult, category):
                if cat != "origin_problem":
                    continue
                if sample_id not in difficult_map:
                    difficult_map[sample_id] = diff
                else:
                    if difficult_map[sample_id] != diff:
                        raise ValueError(
                            f"Conflict: id={sample_id} has multiple difficulties: "
                            f"{difficult_map[sample_id]} vs {diff}"
                        )

            id_to_samples = {}
            for cat, sample_id, score in zip(category, id, scores):
                if sample_id not in id_to_samples:
                    id_to_samples[sample_id] = []
                id_to_samples[sample_id].append(score)

            updates = []
            updates_log = []
            for sample_id, scores in id_to_samples.items():
                total_score = sum(scores)
                sample_count = len(scores)
                avg_score = total_score / sample_count

                old_diff = difficult_map[sample_id]
                if avg_score > 0.8:
                    new_diff = old_diff - 2
                elif avg_score > 0.6:
                    new_diff = old_diff - 1
                elif avg_score < 0.4:
                    new_diff = old_diff + 1
                elif avg_score < 0.2:
                    new_diff = old_diff + 2
                else:
                    new_diff = old_diff

                updates_log.append((sample_id, new_diff, old_diff, avg_score))
                updates.append((sample_id, new_diff))
            updates = list(set(updates))
            updates_log = list(set(updates_log))

            self.train_dataloader.dataset.update_difficulty(updates)

            stats = {
                'new_diff': {},
                'old_diff': {},
                'avg_score': 0.0
            }

            total_entries = 0
            total_score = 0.0
            for uid, new_diff, old_diff, avg_score in updates_log:
                self._log_difficulty_change(uid, new_diff, old_diff, avg_score)
                if new_diff != 0:
                    stats['new_diff'][new_diff] = stats['new_diff'].get(new_diff, 0) + 1
                if old_diff != 0:
                    stats['old_diff'][old_diff] = stats['old_diff'].get(old_diff, 0) + 1
                total_score += avg_score
                total_entries += 1

            if total_entries > 0:
                stats['avg_avg_score'] = float(total_score / total_entries)
            log_path = self.config.trainer.Diffculty_Updates_Path
            log_path = log_path.replace("difficulty_updates", "statistics")
            self._append_to_json_log(log_path, stats)

            def normalize_advantages(advantages):
                mean = advantages.mean()
                std = advantages.std()
                std = std if std > 0 else 1.0
                normalized = (advantages - mean) / std
                return normalized

            def minmax_normalize_advantages(advantages):
                max_abs = torch.max(torch.abs(advantages))
                if max_abs == 0:
                    return advantages
                normalized = advantages / max_abs
                return normalized

            if weight_mode.startswith("weight_before"):
                local_advantages, local_returns = core_algos.compute_grpo_outcome_advantage(
                    token_level_rewards, response_mask, index
                )
                global_advantages, global_returns = core_algos.compute_grpo_outcome_advantage(
                    token_level_rewards, response_mask, global_index
                )
                weighted_global_advantages = []
                for diff_diff, g_adv in zip(difficult_diffs, global_advantages):
                    weighted_global_advantages.append(weighted_advantage(diff_diff, g_adv))
                weighted_global_advantages = torch.stack(weighted_global_advantages)

                if 'rms_norm' in weight_mode:
                    local_advantages = rms_normalize_advantages(local_advantages)  # noqa: F821
                    weighted_global_advantages = rms_normalize_advantages(weighted_global_advantages)  # noqa: F821
                elif 'minmax_norm' in weight_mode:
                    local_advantages = minmax_normalize_advantages(local_advantages)
                    weighted_global_advantages = minmax_normalize_advantages(weighted_global_advantages)
                elif 'zscore_norm' in weight_mode:
                    local_advantages = normalize_advantages(local_advantages)
                    weighted_global_advantages = normalize_advantages(weighted_global_advantages)

                if Adjust_Low_Reward_Local is True:
                    local_advantages = adjust_low_reward_advantages(
                        local_advantages, index, token_level_rewards
                    )

                if Adjust_Low_Reward_Global is True:
                    weighted_global_advantages = adjust_low_reward_advantages(
                        weighted_global_advantages, global_index, token_level_rewards
                    )

                advantages = alpfa * local_advantages + beta * weighted_global_advantages
                returns = alpfa * local_returns + beta * weighted_global_advantages

            elif weight_mode.startswith("weight_after"):
                local_advantages, local_returns = core_algos.compute_grpo_outcome_advantage(
                    token_level_rewards, response_mask, index
                )
                global_advantages, global_returns = core_algos.compute_grpo_outcome_advantage(
                    token_level_rewards, response_mask, global_index
                )
                if 'rms_norm' in weight_mode:
                    local_advantages = rms_normalize_advantages(local_advantages)  # noqa: F821
                    global_advantages = rms_normalize_advantages(global_advantages)  # noqa: F821
                elif 'minmax_norm' in weight_mode:
                    local_advantages = minmax_normalize_advantages(local_advantages)
                    global_advantages = minmax_normalize_advantages(global_advantages)
                elif 'zscore_norm' in weight_mode:
                    local_advantages = normalize_advantages(local_advantages)
                    global_advantages = normalize_advantages(global_advantages)

                weighted_global_advantages = []
                for diff_diff, g_adv in zip(difficult_diffs, global_advantages):
                    weighted_global_advantages.append(weighted_advantage(diff_diff, g_adv))
                weighted_global_advantages = torch.stack(weighted_global_advantages)

                if Adjust_Low_Reward_Local is True:
                    local_advantages = adjust_low_reward_advantages(
                        local_advantages, index, token_level_rewards
                    )

                if Adjust_Low_Reward_Global is True:
                    weighted_global_advantages = adjust_low_reward_advantages(
                        weighted_global_advantages, global_index, token_level_rewards
                    )

                advantages = alpfa * local_advantages + beta * weighted_global_advantages
                returns = alpfa * local_returns + beta * weighted_global_advantages

            elif weight_mode.startswith("weightafter1-5"):
                local_advantages, local_returns = core_algos.compute_grpo_outcome_advantage(
                    token_level_rewards, response_mask, index
                )
                global_advantages, global_returns = core_algos.compute_grpo_outcome_advantage(
                    token_level_rewards, response_mask, global_index
                )
                if 'rms_norm' in weight_mode:
                    local_advantages = rms_normalize_advantages(local_advantages)  # noqa: F821
                    global_advantages = rms_normalize_advantages(global_advantages)  # noqa: F821
                elif 'minmax_norm' in weight_mode:
                    local_advantages = minmax_normalize_advantages(local_advantages)
                    global_advantages = minmax_normalize_advantages(global_advantages)
                elif 'zscore_norm' in weight_mode:
                    local_advantages = normalize_advantages(local_advantages)
                    global_advantages = normalize_advantages(global_advantages)

                weighted_global_advantages = []
                for diff_diff, g_adv in zip(difficult_diffs, global_advantages):
                    weighted_global_advantages.append(weighted_advantage(diff_diff, g_adv))
                weighted_global_advantages = torch.stack(weighted_global_advantages)

                if Adjust_Low_Reward_Local is True:
                    local_advantages = adjust_low_reward_advantages(
                        local_advantages, index, token_level_rewards
                    )
                if Adjust_Low_Reward_Global is True:
                    weighted_global_advantages = adjust_low_reward_advantages(
                        weighted_global_advantages, global_index, token_level_rewards
                    )
                advantages = alpfa * local_advantages + beta * weighted_global_advantages
                returns = alpfa * local_returns + beta * weighted_global_advantages

            elif weight_mode.startswith("weightafter_klcov"):
                ref_log_probs = data.batch["ref_log_probs"]
                local_advantages, local_returns = core_algos.compute_grpo_outcome_advantage_kl_cov(
                    token_level_rewards, response_mask, index, 1e-6, old_log_probs, ref_log_probs, True, 2e-4, 1.0
                )
                global_advantages, global_returns = core_algos.compute_grpo_outcome_advantage_kl_cov(
                    token_level_rewards, response_mask, global_index, 1e-6, old_log_probs, ref_log_probs, True, 2e-4, 1.0
                )
                if 'rms_norm' in weight_mode:
                    local_advantages = rms_normalize_advantages(local_advantages)  # noqa: F821
                    global_advantages = rms_normalize_advantages(global_advantages)  # noqa: F821
                elif 'minmax_norm' in weight_mode:
                    local_advantages = minmax_normalize_advantages(local_advantages)
                    global_advantages = minmax_normalize_advantages(global_advantages)
                elif 'zscore_norm' in weight_mode:
                    local_advantages = normalize_advantages(local_advantages)
                    global_advantages = normalize_advantages(global_advantages)

                weighted_global_advantages = []
                for diff_diff, g_adv in zip(difficult_diffs, global_advantages):
                    weighted_global_advantages.append(weighted_advantage(diff_diff, g_adv))
                weighted_global_advantages = torch.stack(weighted_global_advantages)

                if Adjust_Low_Reward_Local is True:
                    local_advantages = adjust_low_reward_advantages(
                        local_advantages, index, token_level_rewards
                    )
                if Adjust_Low_Reward_Global is True:
                    weighted_global_advantages = adjust_low_reward_advantages(
                        weighted_global_advantages, global_index, token_level_rewards
                    )
                advantages = alpfa * local_advantages + beta * weighted_global_advantages
                returns = alpfa * local_returns + beta * weighted_global_advantages
                advantages = advantages / 5
                returns = returns / 5

            elif weight_mode.startswith("no_weight"):
                local_advantages, local_returns = core_algos.compute_grpo_outcome_advantage(
                    token_level_rewards, response_mask, index
                )
                global_advantages, global_returns = core_algos.compute_grpo_outcome_advantage(
                    token_level_rewards, response_mask, global_index
                )
                if 'rms_norm' in weight_mode:
                    local_advantages = rms_normalize_advantages(local_advantages)  # noqa: F821
                    weighted_global_advantages = rms_normalize_advantages(global_advantages)  # noqa: F821
                elif 'minmax_norm' in weight_mode:
                    local_advantages = minmax_normalize_advantages(local_advantages)
                    weighted_global_advantages = minmax_normalize_advantages(global_advantages)
                elif 'zscore_norm' in weight_mode:
                    local_advantages = normalize_advantages(local_advantages)
                    weighted_global_advantages = normalize_advantages(global_advantages)
                else:
                    weighted_global_advantages = global_advantages

                if Adjust_Low_Reward_Local is True:
                    local_advantages = adjust_low_reward_advantages(
                        local_advantages, index, token_level_rewards
                    )

                if Adjust_Low_Reward_Global is True:
                    weighted_global_advantages = adjust_low_reward_advantages(
                        weighted_global_advantages, global_index, token_level_rewards
                    )

                advantages = alpfa * local_advantages + beta * weighted_global_advantages
                returns = alpfa * local_returns + beta * weighted_global_advantages

            self._save_full_vectors_to_json(
                data, local_advantages, global_advantages, weighted_global_advantages, advantages, token_level_rewards
            )

        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        return data

    def _compute_advantage(
        self,
        data: DataProto,
        adv_estimator: AdvantageEstimator,
        gamma: float = 1.0,
        lam: float = 1.0,
        Adjust_Low_Reward: bool = False
    ):
        """Compute advantages and returns with multiple estimation methods."""
        token_level_rewards = data.batch["token_level_rewards"]
        response_mask = data.batch["response_mask"]
        index = data.non_tensor_batch["uid"]

        if adv_estimator == AdvantageEstimator.GAE:
            values = data.batch["values"]
            advantages, returns = core_algos.compute_gae_advantage_return(
                token_level_rewards, values, response_mask, gamma, lam
            )
        elif adv_estimator == AdvantageEstimator.GRPO:
            if not Adjust_Low_Reward:
                advantages, returns = core_algos.compute_grpo_outcome_advantage(
                    token_level_rewards, response_mask, index
                )
                self._save_full_vectors_to_json_origin(data, advantages, token_level_rewards)
            else:
                advantages, returns = core_algos.compute_grpo_outcome_advantage(
                    token_level_rewards, response_mask, index
                )
                advantages = adjust_low_reward_advantages(advantages, index, token_level_rewards)
                self._save_full_vectors_to_json_origin(data, advantages, token_level_rewards)

        elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
            advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
                token_level_rewards, response_mask, gamma
            )

        elif adv_estimator == AdvantageEstimator.REMAX:
            reward_baselines = data.batch["reward_baselines"]
            advantages, returns = core_algos.compute_remax_outcome_advantage(
                token_level_rewards, reward_baselines, response_mask
            )

        elif adv_estimator == AdvantageEstimator.RLOO:
            advantages, returns = core_algos.compute_rloo_outcome_advantage(
                token_level_rewards, response_mask, index
            )
        else:
            raise NotImplementedError(f"Unsupported advantage estimator: {adv_estimator}")

        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        return data

    def _log_difficulty_change(self, id_, new_diff, old_diff, math_reward):
        """Append a difficulty change record in JSONL format."""
        log_entry = {
            "id": id_,
            "old_difficulty": float(old_diff),
            "new_difficulty": float(new_diff),
            "math_reward": float(math_reward)
        }
        log_path = self.config.trainer.Diffculty_Updates_Path
        self._append_to_json_log(log_path, log_entry)

    def _append_to_json_log(self, filename, data):
        """Append a JSON object per line to a file with advisory locking."""
        try:
            with open(filename, "a+") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                f.seek(0, 2)
                json.dump(data, f)
                f.write("\n")
                fcntl.flock(f, fcntl.LOCK_UN)
        except Exception as e:
            print(f"Error writing to {filename}: {str(e)}")

    def _save_full_vectors_to_json(self, data, local_advantages, global_advantages,
                                   adj_global_advantages, advtanges, token_level_rewards):
        """
        Save processed vectors to a JSONL file (local/global advantages and token-level reward scores).
        Only the first element per-row is kept for advantage tensors; token rewards are summed.
        """
        output_path = self.config.trainer.Full_Vector_Data_Path
        existing_lines = []
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                existing_lines = [line.strip() for line in f if line.strip()]

        def to_numpy(x):
            return x.cpu().numpy() if hasattr(x, 'cpu') else np.array(x)

        local_adv = to_numpy(local_advantages)[:, 0]
        global_adv = to_numpy(global_advantages)[:, 0]
        adj_global_adv = to_numpy(adj_global_advantages)[:, 0]
        adv = to_numpy(advtanges)[:, 0]

        token_rewards = to_numpy(token_level_rewards)
        scores = token_rewards.sum(axis=-1)

        jsonl_data = [
            json.dumps({
                "id": str(data.non_tensor_batch["id"][i]),
                "local_idx": str(data.non_tensor_batch["uid"][i]),
                "global_idx": str(data.non_tensor_batch["global_uid"][i]),
                "local_advantage": float(local_adv[i]),
                "global_advantage": float(global_adv[i]),
                "adj_global_advantage": float(adj_global_adv[i]),
                "advantage": float(adv[i]),
                "token_rewards_score": float(scores[i]),
                "problem": str(data.non_tensor_batch["problem"][i]),
                "index": i,
                "category": data.non_tensor_batch["category"][i],
                "difficulty": data.non_tensor_batch["difficulty"][i]
            })
            for i in range(len(data.non_tensor_batch["id"]))
        ]

        with open(output_path, "a", encoding="utf-8") as f:
            for line in jsonl_data:
                f.write(line + "\n")

        print(f"Saved full vector data: added {len(jsonl_data)} samples, total {len(existing_lines) + len(jsonl_data)}")

    def _save_full_vectors_to_json_origin(self, data, local_advantages, token_level_rewards):
        """
        Save processed vectors (local advantages and token reward scores) to a JSONL file.
        Only the first element per-row is kept for advantage tensors; token rewards are summed.
        """
        output_path = self.config.trainer.Full_Vector_Data_Path
        existing_lines = []
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                existing_lines = [line.strip() for line in f if line.strip()]

        def to_numpy(x):
            return x.cpu().numpy() if hasattr(x, 'cpu') else np.array(x)

        local_adv = to_numpy(local_advantages)[:, 0]
        token_rewards = to_numpy(token_level_rewards)
        scores = token_rewards.sum(axis=-1)

        jsonl_data = [
            json.dumps({
                "id": str(data.non_tensor_batch["id"][i]),
                "local_advantage": float(local_adv[i]),
                "token_rewards_score": float(scores[i]),
                "problem": str(data.non_tensor_batch["problem"][i]),
                "index": i,
            })
            for i in range(len(data.non_tensor_batch["id"]))
        ]

        with open(output_path, "a", encoding="utf-8") as f:
            for line in jsonl_data:
                f.write(line + "\n")

        print(f"Saved full vector data: added {len(jsonl_data)} samples, total {len(existing_lines) + len(jsonl_data)}")

    def fit(self):
        """
        PPO training loop executed on the driver.
        """
        self.logger = Tracker(loggers=self.config.trainer.logger, config=self.config.to_dict())
        self.global_step = 0
        main_tqdm = tqdm(range(self.training_steps), desc="Running step", position=0)
        val_metrics: Optional[Dict[str, Any]] = None

        self._load_checkpoint()
        main_tqdm.update(self.global_step)

        if self.val_reward_fn is not None and self.config.trainer.val_before_train:
            val_metrics = self._validate()
            self.logger.log(data=val_metrics, step=self.global_step)
            if self.config.trainer.val_only:
                return

        self.data_iterator = iter(self.train_dataloader)

        while self.global_step < self.training_steps:
            self.global_step += 1
            metrics, timing_raw = {}, {}

            with timer("step", timing_raw):
                with timer("gen", timing_raw):
                    self.actor_rollout_ref_wg.prepare_rollout_engine()
                    batch = self._make_batch_data(metrics=metrics)
                    self.actor_rollout_ref_wg.release_rollout_engine()

                self._balance_batch(batch, metrics=metrics)
                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                if "token_level_scores" not in batch.batch:
                    if self.config.algorithm.Curriculum_Reward:
                        with timer("reward", timing_raw):
                            reward_ref = self.reward_fn.compute_reward.remote(batch)
                    else:
                        with timer("reward", timing_raw):
                            reward_ref = self.reward_fn.compute_reward.remote(batch)

                with timer("old", timing_raw):
                    old_log_probs = self.actor_rollout_ref_wg.compute_log_probs(batch)
                    batch = batch.union(old_log_probs)

                if self.use_reference_policy:
                    with timer("ref", timing_raw):
                        ref_log_probs = self.actor_rollout_ref_wg.compute_ref_log_probs(batch)
                        batch = batch.union(ref_log_probs)

                if self.use_critic:
                    with timer("values", timing_raw):
                        values = self.critic_wg.compute_values(batch)
                        batch = batch.union(values)

                with timer("adv", timing_raw):
                    if "token_level_scores" not in batch.batch:
                        reward_tensor, reward_metrics = ray.get(reward_ref)
                        batch.batch["token_level_scores"] = reward_tensor
                        reward_metrics = {f"reward/{k}": v for k, v in reduce_metrics(reward_metrics).items()}
                        metrics.update(reward_metrics)

                    if not self.config.algorithm.use_kl_loss and self.use_reference_policy:
                        batch, kl_metrics = apply_kl_penalty(batch, self.kl_ctrl, self.config.algorithm.kl_penalty)
                        metrics.update(kl_metrics)
                    else:
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                    if self.config.trainer.Difficulty_Adaptation and self.config.trainer.Difficulty_Change:
                        batch = self._compute_difficult_adaption_advantage_difficulty(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            weight_mode=self.config.algorithm.weight_mode,
                            Adjust_Low_Reward_Local=self.config.algorithm.Adjust_Low_Reward_Local,
                            Adjust_Low_Reward_Global=self.config.algorithm.Adjust_Low_Reward_Global
                        )
                    elif self.config.trainer.Difficulty_Adaptation:
                        batch = self._compute_difficult_adaption_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                        )
                    else:
                        batch = self._compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            Adjust_Low_Reward=self.config.algorithm.Adjust_Low_Reward_Local
                        )

                if self.use_critic:
                    with timer("update_critic", timing_raw):
                        critic_output = self.critic_wg.update_critic(batch)
                    critic_metrics = reduce_metrics(critic_output.non_tensor_batch)
                    metrics.update(critic_metrics)

                if self.config.trainer.critic_warmup <= self.global_step:
                    with timer("update_actor", timing_raw):
                        actor_output = self.actor_rollout_ref_wg.update_actor(batch)
                    actor_metrics = reduce_metrics(actor_output.non_tensor_batch)
                    metrics.update(actor_metrics)

                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.val_freq > 0
                    and self.global_step % self.config.trainer.val_freq == 0
                ):
                    with timer("validation", timing_raw):
                        val_metrics = self._validate()
                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and self.global_step % self.config.trainer.save_freq == 0:
                    with timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()

            num_gpus = self.resource_pool_manager.get_num_gpus()
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, num_gpus=num_gpus))

            self.logger.log(data=metrics, step=self.global_step)
            main_tqdm.update()

        if self.val_reward_fn is not None:
            if (
                val_metrics is None
                or self.config.trainer.val_freq <= 0
                or self.global_step % self.config.trainer.val_freq != 0
            ):
                val_metrics = self._validate()
                self.logger.log(data=val_metrics, step=self.global_step)
            print(f"Final validation metrics: {convert_dict_to_str(val_metrics)}")

        if self.config.trainer.save_freq <= 0 or self.global_step % self.config.trainer.save_freq != 0:
            self._save_checkpoint()

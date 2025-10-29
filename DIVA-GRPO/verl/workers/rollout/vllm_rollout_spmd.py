# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.distributed
from tensordict import TensorDict
from transformers import PreTrainedTokenizer, ProcessorMixin
from vllm import LLM, RequestOutput, SamplingParams

from ...protocol import DataProto
from ...utils import torch_functional as VF
from ...utils.dataset import process_image, process_video
from ...utils.torch_dtypes import PrecisionType
from .base import BaseRollout
from .config import RolloutConfig


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, np.ndarray]:
    # repeat the elements, supports both tensor and numpy array
    # 直接复制样本，扩展batch size到repeat次数
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


def _get_logit_bias(processor: Optional[ProcessorMixin]) -> Optional[Dict[int, float]]:
    # enforce vllm to not output image token
    # TODO: add video token
    if processor is not None and hasattr(processor, "image_token"):
        image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        return {image_token_id: -100}
    else:
        return None


def _process_multi_modal_data(
    multi_modal_data: Dict[str, Any], min_pixels: int, max_pixels: int, video_fps: float
) -> Dict[str, Any]:
    # may convert image path to image object
    images, videos = [], []
    if "images" in multi_modal_data:
        if isinstance(multi_modal_data["images"],dict):
            images.append(process_image(multi_modal_data["images"], min_pixels, max_pixels))
        else:
            for image in multi_modal_data["images"]:
                images.append(process_image(image, min_pixels, max_pixels))

    if "videos" in multi_modal_data:
        for video in multi_modal_data["videos"]:
            videos.append(process_video(video, min_pixels, max_pixels, video_fps))

    if len(images) != 0:
        return {"image": images}

    if len(videos) != 0:
        return {"video": videos}

    return None


class vLLMRollout(BaseRollout):
    def __init__(
        self,
        model_path: str,
        config: RolloutConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
    ):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
        """
        super().__init__()
        self.rank = int(os.getenv("RANK", "0"))
        self.config = config
        self.pad_token_id = tokenizer.pad_token_id
        self.use_tqdm = (self.rank == 0) and (not config.disable_tqdm)
        if config.tensor_parallel_size > torch.distributed.get_world_size():
            raise ValueError("Tensor parallelism size should be less than world size.")

        if config.max_num_batched_tokens < config.prompt_length + config.response_length:
            raise ValueError("max_num_batched_tokens should be greater than prompt_length + response_length.")

        engine_kwargs = {}
        if processor is not None:  # only VLMs have processor
            engine_kwargs["disable_mm_preprocessor_cache"] = True
            if config.limit_images:
                engine_kwargs["limit_mm_per_prompt"] = {"image": config.limit_images}

        self.inference_engine = LLM(
            model=model_path,
            skip_tokenizer_init=False,
            trust_remote_code=config.trust_remote_code,
            load_format="dummy",
            dtype=PrecisionType.to_str(PrecisionType.to_dtype(config.dtype)),
            seed=config.seed,
            max_model_len=config.max_model_len or config.prompt_length + config.response_length,
            distributed_executor_backend="external_launcher",
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_num_batched_tokens=config.max_num_batched_tokens,
            disable_log_stats=config.disable_log_stats,
            enforce_eager=config.enforce_eager,
            disable_custom_all_reduce=True,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_sleep_mode=True,
            **engine_kwargs,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        sampling_kwargs = {
            "max_tokens": config.response_length,
            "detokenize": False,
            "logit_bias": _get_logit_bias(processor),
        }
        default_sampling_params = SamplingParams()
        for key in config.to_dict().keys():
            if hasattr(default_sampling_params, key):
                sampling_kwargs[key] = getattr(config, key)

        print(f"Sampling params: {sampling_kwargs}.")
        self.sampling_params = SamplingParams(**sampling_kwargs)

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)

        yield
        # roll back to previous sampling params
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    # 禁用梯度计算，提高推理效率并减少内存占用
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """
        根据给定的提示(prompts)生成文本序列
        
        参数:
            prompts: DataProto对象，包含:
                - batch: 包含输入张量的字典
                - non_tensor_batch: 非张量数据
                - meta_info: 生成参数和配置信息
        
        返回:
            DataProto: 包含生成的序列和相关信息的对象
        """
        
        # ==================== 输入数据准备 ====================
        # 从prompts中提取输入张量
        # input_ids: 提示文本的token ID (batch_size x 提示长度)
        input_ids: torch.Tensor = prompts.batch["input_ids"]  
        # attention_mask: 注意力掩码，标记哪些token需要被关注(左填充)
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        # position_ids: 位置编码，表示每个token的位置信息
        position_ids: torch.Tensor = prompts.batch["position_ids"]
        # eos_token_id: 结束符token ID，用于停止生成
        eos_token_id: int = prompts.meta_info["eos_token_id"]
        batch_size = input_ids.size(0)  # 获取当前批次大小

        # 提取非张量数据
        non_tensor_batch = prompts.non_tensor_batch
        # raw_prompt_ids: 原始提示文本的token ID(未经处理的)
        batch_raw_prompt_ids = non_tensor_batch.pop("raw_prompt_ids")
        # multi_modal_data: 多模态数据(如图像/视频等)，可选
        batch_multi_modal_data = non_tensor_batch.pop("multi_modal_data", None)
        
        # 检查批次大小是否一致(防止vLLM分片管理器工作异常)
        if batch_size != len(batch_raw_prompt_ids):
            raise RuntimeError("vllm sharding manager is not work properly.")

        # ==================== 准备vLLM引擎输入 ====================
        if batch_multi_modal_data is not None:
            # 如果有多模态数据，需要将每个样本与其对应的多模态数据一起处理
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(batch_raw_prompt_ids, batch_multi_modal_data):
                vllm_inputs.append(
                    {
                        "prompt_token_ids": list(raw_prompt_ids),  # 原始提示token ID
                        "multi_modal_data": _process_multi_modal_data(  # 处理多模态数据
                            multi_modal_data,
                            prompts.meta_info["min_pixels"],  # 最小像素值
                            prompts.meta_info["max_pixels"],  # 最大像素值
                            prompts.meta_info["video_fps"],    # 视频帧率
                        ),
                    }
                )
        else:
            # 纯文本输入，只需使用提示token ID
            vllm_inputs = [{"prompt_token_ids": list(raw_prompt_ids)} for raw_prompt_ids in batch_raw_prompt_ids]

        # ==================== 序列生成 ====================
        # 使用自定义的采样参数(从prompts.meta_info中获取)
        with self.update_sampling_params(**prompts.meta_info):
            # 调用推理引擎生成文本，在此按照采样次数获取推理多次的结果
            completions: List[RequestOutput] = self.inference_engine.generate(
                prompts=vllm_inputs,          # 输入数据
                sampling_params=self.sampling_params,  # 采样参数
                use_tqdm=self.use_tqdm        # 是否显示进度条
            )
            
            # 从输出中提取生成的token ID
            response_ids = [output.token_ids for completion in completions for output in completion.outputs]
            # 将生成的响应填充到固定长度(使用pad_token_id填充)
            response_ids = VF.pad_2d_list_to_length(
                response_ids, 
                self.pad_token_id, 
                max_length=self.config.response_length
            ).to(input_ids.device)  # 确保与输入在同一个设备上

            # 调整除输出外的其他所有参数，以匹配输出的格式size
            # 处理每个提示生成多个样本的情况(n > 1)
            if self.sampling_params.n > 1:
                # 调整批次大小并重复输入数据
                batch_size = batch_size * self.sampling_params.n
                input_ids = _repeat_interleave(input_ids, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                if batch_multi_modal_data is not None:
                    batch_multi_modal_data = _repeat_interleave(batch_multi_modal_data, self.sampling_params.n)

        # ==================== 后处理 ====================
        # 合并提示和生成的token ID
        sequence_ids = torch.cat([input_ids, response_ids], dim=-1)
        response_length = response_ids.size(1)  # 获取生成文本的长度
        
        # 计算生成部分的位置ID增量
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
        
        # 特殊处理: 针对qwen2vl等多查询旋转位置编码模型
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # 更新完整序列(提示+生成)的位置ID
        # 生成部分的位置ID = 提示最后位置 + 增量
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1 | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3 | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        
        # 创建响应掩码(1表示真实token，0表示填充)
        response_mask = VF.get_response_mask(
            response_ids=response_ids, 
            eos_token_id=eos_token_id, 
            dtype=attention_mask.dtype
        )
        # 合并提示和响应的注意力掩码
        attention_mask = torch.cat((attention_mask, response_mask), dim=-1)

        # ==================== 准备输出 ====================
        # 构建包含所有相关张量的批次数据
        batch = TensorDict(
            {
                "prompts": input_ids,         # 原始提示
                "responses": response_ids,    # 生成的响应
                "input_ids": sequence_ids,    # 完整序列(提示+响应)
                "attention_mask": attention_mask,  # 合并后的注意力掩码
                "response_mask": response_mask,    # 仅响应部分的掩码
                "position_ids": position_ids,      # 完整的位置ID
            },
            batch_size=batch_size,
        )
        
        # 如果有多模态数据，包含在输出中
        if batch_multi_modal_data is not None:
            non_tensor_batch = {"multi_modal_data": batch_multi_modal_data}
        else:
            non_tensor_batch = {}

        # 返回包含所有数据的DataProto对象
        return DataProto(
            batch=batch, 
            non_tensor_batch=non_tensor_batch, 
            meta_info=prompts.meta_info
        )

    # @torch.no_grad()
    # def generate_sequences(self, prompts: DataProto) -> DataProto:
    #     # left-padded attention_mask
    #     input_ids: torch.Tensor = prompts.batch["input_ids"]  # (bs, prompt_length)
    #     attention_mask: torch.Tensor = prompts.batch["attention_mask"]
    #     position_ids: torch.Tensor = prompts.batch["position_ids"]
    #     eos_token_id: int = prompts.meta_info["eos_token_id"]
    #     batch_size = input_ids.size(0)

    #     non_tensor_batch = prompts.non_tensor_batch
    #     batch_raw_prompt_ids = non_tensor_batch.pop("raw_prompt_ids")
    #     batch_multi_modal_data = non_tensor_batch.pop("multi_modal_data", None)
    #     if batch_size != len(batch_raw_prompt_ids):
    #         raise RuntimeError("vllm sharding manager is not work properly.")

    #     if batch_multi_modal_data is not None:
    #         vllm_inputs = []
    #         for raw_prompt_ids, multi_modal_data in zip(batch_raw_prompt_ids, batch_multi_modal_data):
    #             vllm_inputs.append(
    #                 {
    #                     "prompt_token_ids": list(raw_prompt_ids),
    #                     "multi_modal_data": _process_multi_modal_data(
    #                         multi_modal_data,
    #                         prompts.meta_info["min_pixels"],
    #                         prompts.meta_info["max_pixels"],
    #                         prompts.meta_info["video_fps"],
    #                     ),
    #                 }
    #             )
    #     else:
    #         vllm_inputs = [{"prompt_token_ids": list(raw_prompt_ids)} for raw_prompt_ids in batch_raw_prompt_ids]

    #     # users can customize different sampling_params at different run
    #     with self.update_sampling_params(**prompts.meta_info):
    #         completions: List[RequestOutput] = self.inference_engine.generate(
    #             prompts=vllm_inputs, sampling_params=self.sampling_params, use_tqdm=self.use_tqdm
    #         )
    #         response_ids = [output.token_ids for completion in completions for output in completion.outputs]
    #         response_ids = VF.pad_2d_list_to_length(
    #             response_ids, self.pad_token_id, max_length=self.config.response_length
    #         ).to(input_ids.device)

    #         if self.sampling_params.n > 1:
    #             batch_size = batch_size * self.sampling_params.n
    #             input_ids = _repeat_interleave(input_ids, self.sampling_params.n)
    #             attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
    #             position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
    #             if batch_multi_modal_data is not None:
    #                 batch_multi_modal_data = _repeat_interleave(batch_multi_modal_data, self.sampling_params.n)

    #     sequence_ids = torch.cat([input_ids, response_ids], dim=-1)
    #     response_length = response_ids.size(1)
    #     delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
    #     delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
    #     if position_ids.dim() == 3:  # qwen2vl mrope
    #         delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

    #     # prompt: left pad + response: right pad
    #     # attention_mask: [0,0,0,0,1,1,1,1 | 1,1,1,0,0,0,0,0]
    #     # position_ids:   [0,0,0,0,0,1,2,3 | 4,5,6,7,8,9,10,11]
    #     response_position_ids = position_ids[..., -1:] + delta_position_id
    #     position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
    #     response_mask = VF.get_response_mask(
    #         response_ids=response_ids, eos_token_id=eos_token_id, dtype=attention_mask.dtype
    #     )
    #     attention_mask = torch.cat((attention_mask, response_mask), dim=-1)

    #     # all the tp ranks should contain the same data here. data in all ranks are valid
    #     batch = TensorDict(
    #         {
    #             "prompts": input_ids,
    #             "responses": response_ids,
    #             "input_ids": sequence_ids,  # here input_ids become the whole sentences
    #             "attention_mask": attention_mask,
    #             "response_mask": response_mask,
    #             "position_ids": position_ids,
    #         },
    #         batch_size=batch_size,
    #     )
    #     if batch_multi_modal_data is not None:
    #         non_tensor_batch = {"multi_modal_data": batch_multi_modal_data}
    #     else:
    #         non_tensor_batch = {}

    #     return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=prompts.meta_info)

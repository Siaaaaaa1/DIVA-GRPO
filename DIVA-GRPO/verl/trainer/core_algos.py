# Copyright 2022 The HuggingFace Team
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
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum
from typing import TYPE_CHECKING, Dict, Literal, Tuple, Optional
from typing import List, Any
import numpy as np
import torch
import torch.nn.functional as F
from ..utils import torch_functional as VF


if TYPE_CHECKING:
    from .config import AlgorithmConfig


class KLController(ABC):
    kl_coef: float
    """KL coefficient."""

    @abstractmethod
    def update(self, current_kl: float, n_steps: int):
        """Update kl_coef according to current KL."""
        ...


class AdaptiveKLController(KLController):
    """Adaptive KL controller described in: https://arxiv.org/pdf/1909.08593.pdf

    Copied from https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/utils.py#L54"""

    def __init__(self, init_kl_coef: float, target_kl: float, horizon: float):
        self.kl_coef = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl: float, n_steps: int):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.kl_coef *= mult


class FixedKLController(KLController):
    """Fixed KL controller.

    Copeid from https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/utils.py#L72"""

    def __init__(self, init_kl_coef: float):
        self.kl_coef = init_kl_coef

    def update(self, current_kl: float, n_steps: int):
        pass


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REMAX = "remax"
    RLOO = "rloo"


def get_kl_controller(algorithm_config: "AlgorithmConfig") -> KLController:
    """Adapted from https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L319"""
    if algorithm_config.kl_type == "fixed":
        kl_ctrl = FixedKLController(init_kl_coef=algorithm_config.kl_coef)
    elif algorithm_config.kl_type == "adaptive":
        assert algorithm_config.kl_horizon > 0, f"horizon must be larger than 0. Got {algorithm_config.kl_horizon}."
        kl_ctrl = AdaptiveKLController(
            init_kl_coef=algorithm_config.kl_coef,
            target_kl=algorithm_config.kl_target,
            horizon=algorithm_config.kl_horizon,
        )
    else:
        raise ValueError(f"Unknown kl type: {algorithm_config.kl_type}.")

    return kl_ctrl


@torch.no_grad()
def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Adapted from https://github.com/huggingface/trl/blob/v0.16.0/trl/trainer/ppo_trainer.py#L513

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length). The token after eos tokens have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    lastgaelam = 0
    advantages_reversed = []
    gen_len = token_level_rewards.shape[-1]
    for t in reversed(range(gen_len)):
        nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
        delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
        lastgaelam = delta + gamma * lam * lastgaelam
        advantages_reversed.append(lastgaelam)

    advantages = torch.stack(advantages_reversed[::-1], dim=1)
    returns = advantages + values
    advantages = VF.masked_whiten(advantages, response_mask)
    return advantages, returns

def compute_policy_loss_clip_cov(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    loss_agg_mode="token-mean",
    clip_ratio=0.0002,
    clip_cov_lb=1.0,
    clip_cov_ub=5.0,
):
    """
    Compute the clipped policy objective and related metrics for Clip-Cov.
    Adapted from
    https://github.com/PRIME-RL/Entropy-Mechanism-of-RL/blob/main/verl/trainer/ppo/core_algos.py
    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        clip_ratio (float, optional):
            Ratio for clipping the covariance. Defaults to 0.0002.
        clip_cov_lb (float, optional):
            Lower bound for clipping covariance. Defaults to 1.0.
        clip_cov_ub (float, optional):
            Upper bound for clipping covariance. Defaults to 5.0.
    """
    assert clip_ratio > 0, "clip_ratio should be larger than 0."
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio

    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange

    corr = torch.ones_like(advantages)
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
    clip_by_origin = (pg_losses2 > pg_losses1) & (response_mask > 0)

    cov_all = (advantages- verl_F.masked_mean(advantages, response_mask)) * (log_prob- verl_F.masked_mean(log_prob.detach(), response_mask))
    cov_all[response_mask == 0] = -torch.inf
    cov_all[clip_by_origin] = -torch.inf

    clip_num = max(int(clip_ratio * response_mask.sum().item()), 1)
    top_k_idx = (cov_all < clip_cov_ub) & (cov_all > clip_cov_lb) & (response_mask > 0)
    top_k_idx = torch.nonzero(top_k_idx)

    if len(top_k_idx) > 0:
        perm = torch.randperm(len(top_k_idx))
        top_k_idx = top_k_idx[perm[:min(clip_num, len(top_k_idx))]]
    else:
        top_k_idx = torch.empty((0, 2), device=cov_all.device, dtype=torch.long)

    corr[top_k_idx[:, 0], top_k_idx[:, 1]] = 0

    pg_clipfrac = verl_F.masked_mean((corr==0).float(), response_mask)

    pg_losses = torch.maximum(pg_losses1, pg_losses2) * corr
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, torch.tensor(0.)


from typing import Tuple, List, Any, Optional
from collections import defaultdict
import torch

def compute_policy_loss_kl_cov(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    loss_agg_mode="token-mean",
    k_ratio=0.0002,
    ppo_kl_coef=1,
):
    """
    Compute the clipped policy objective and related metrics for Clip-Cov.

    Args:
        old_log_prob (torch.Tensor): Log-probabilities of actions under the old policy, shape (B, T).
        log_prob (torch.Tensor): Log-probabilities of actions under the current policy, shape (B, T).
        advantages (torch.Tensor): Advantage estimates for each action, shape (B, T).
        response_mask (torch.Tensor): Mask indicating which tokens to include, shape (B, T).
        loss_agg_mode (str): Aggregation mode for `agg_loss`. Defaults to "token-mean".
        k_ratio (float): Ratio for selecting the top-k covariance values. Defaults to 0.0002.
        ppo_kl_coef (float): Coefficient for the KL penalty term in the loss. Defaults to 1.
    """
    assert k_ratio > 0, "k_ratio should be larger than 0."
    negative_approx_kl = log_prob - old_log_prob
    abs_kl = negative_approx_kl.abs()
    ratio = torch.exp(negative_approx_kl)
    ppo_kl_abs = verl_F.masked_mean(negative_approx_kl.abs(), response_mask)
    pg_losses1 = -advantages * ratio
    pg_losses_kl = -advantages * ratio + ppo_kl_coef * abs_kl
    pg_losses = pg_losses1

    all_valid = (response_mask > 0)
    all_valid_idx = torch.nonzero(all_valid.reshape(-1), as_tuple=True)[0]
    all_valid_adv = advantages[all_valid].detach().reshape(-1).cpu()
    all_valid_logp = log_prob[all_valid].detach().reshape(-1).cpu()

    k = min(k_ratio, len(all_valid_adv))

    if k != 0:
        cov_lst_all = (all_valid_adv - all_valid_adv.mean()) * (all_valid_logp - all_valid_logp.mean())
        k_percent_nums = max(1, int(len(cov_lst_all) * k_ratio))
        large_cov_idxs = torch.topk(cov_lst_all, k_percent_nums, largest=True).indices

        if len(large_cov_idxs) != 0:
            large_cov_idxs = all_valid_idx[large_cov_idxs]
            pg_losses[large_cov_idxs // advantages.shape[1], large_cov_idxs % advantages.shape[1]] = pg_losses_kl[large_cov_idxs // advantages.shape[1], large_cov_idxs % advantages.shape[1]]

    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, torch.tensor(0.), ppo_kl_abs, torch.tensor(0.)


@torch.no_grad()
def compute_grpo_outcome_advantage_kl_cov(
    token_level_rewards: torch.Tensor,   # (B, T)
    response_mask: torch.Tensor,         # (B, T)
    index: torch.Tensor,                 # (B,)
    eps: float = 1e-6,
    log_prob: Optional[torch.Tensor] = None,      # (B, T)
    old_log_prob: Optional[torch.Tensor] = None,  # (B, T)
    apply_cov_kl: bool = False,
    k_ratio: float = 2e-4,
    ppo_kl_coef: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        advantages: (B, T) — If `apply_cov_kl` is True, subtracts a KL penalty (ppo_kl_coef*|Δlogπ|)
                    on tokens selected by the top-k covariance rule.
        returns:    (B, T) — The GRPO Z-score baseline (without KL).
    """
    assert k_ratio > 0, "k_ratio should be larger than 0."
    device = token_level_rewards.device
    dtype = token_level_rewards.dtype

    scores = token_level_rewards.sum(dim=-1)  # (B,)

    id2score = defaultdict(list)
    bsz = scores.shape[0]
    for i in range(bsz):
        id2score[index[i]].append(scores[i])

    id2mean, id2std = {}, {}
    for gid, lst in id2score.items():
        assert len(lst) > 1, "GRPO requires more than 1 sample per group."
        stack = torch.stack(lst)
        id2mean[gid] = stack.mean()
        id2std[gid]  = stack.std(unbiased=True)

    scores_z = scores.clone()
    for i in range(bsz):
        gid = index[i]
        scores_z[i] = (scores[i] - id2mean[gid]) / (id2std[gid] + eps)

    returns = scores_z.unsqueeze(-1) * response_mask  # (B, T)
    advantages = returns.clone()

    if apply_cov_kl and (log_prob is not None) and (old_log_prob is not None):
        delta_logp = (log_prob - old_log_prob)
        abs_kl = delta_logp.abs()

        valid = (response_mask > 0)
        if valid.any():
            B, T = advantages.shape
            flat_valid_idx = torch.nonzero(valid.reshape(-1), as_tuple=True)[0].to(device)

            adv_flat  = advantages[valid].detach().reshape(-1)
            logp_flat = log_prob[valid].detach().reshape(-1)

            adv_c  = adv_flat - adv_flat.mean()
            logp_c = logp_flat - logp_flat.mean()
            score = adv_c * logp_c

            k_num = max(1, int(score.numel() * k_ratio))
            k_num = min(k_num, score.numel())

            if k_num > 0:
                topk_local = torch.topk(score, k_num, largest=True).indices
                topk_flat = flat_valid_idx[topk_local]

                rows = (topk_flat // T).long()
                cols = (topk_flat %  T).long()

                advantages[rows, cols] = advantages[rows, cols] - ppo_kl_coef * abs_kl[rows, cols]

    return advantages.to(dtype=dtype, device=device), returns.to(dtype=dtype, device=device)


@torch.no_grad()
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,  # (B, T)
    response_mask: torch.Tensor,        # (B, T)
    index: torch.Tensor,                # (B,)
    eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute GRPO advantages for outcome supervision:
      1) Sum token-level rewards per response.
      2) Group by `index` and compute mean/std (group size must be > 1).
      3) Z-score normalize per response using its group's stats.
      4) Expand to token level and apply mask.

    Returns:
        advantages: (B, T)
        returns:    (B, T)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    bsz = scores.shape[0]
    for i in range(bsz):
        id2score[index[i]].append(scores[i])

    for idx_key in id2score:
        assert len(id2score[idx_key]) > 1, "GRPO requires more than 1 sample per group."
        stacked = torch.stack(id2score[idx_key])
        id2mean[idx_key] = torch.mean(stacked)
        id2std[idx_key] = torch.std(stacked, unbiased=True)

    for i in range(bsz):
        scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + eps)

    returns = scores.unsqueeze(-1) * response_mask
    return returns, returns


def multiplier(difficult, advantage):
    m = torch.ones_like(advantage)
    mask_pos = advantage > 0
    mask_neg = advantage < 0

    if difficult < 0:
        m[mask_pos] = 0.5 + 0.5 * (difficult + 9) / 9
        m[mask_neg] = 2.0 - 1.0 * (difficult + 9) / 9
    elif difficult > 0:
        m[mask_pos] = 1.0 + 1.0 * difficult / 9
        m[mask_neg] = 1.0 - 0.5 * difficult / 9
    else:
        m[mask_pos] = 1.0
        m[mask_neg] = 1.0

    return m


@torch.no_grad()
def compute_grpo_outcome_advantage_weight_before_norm(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: torch.Tensor,
    difficult_list: List[Any],
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute GRPO advantages (outcome-based Z-score), applying difficulty weighting before normalization."""
    def multiplier_scores(difficult, score):
        if difficult < 0 and score > 0:
            return 0.5 + 0.5 * (difficult + 9) / 9
        elif difficult < 0 and score < 0:
            return 2.0 - 1.0 * (difficult + 9) / 9
        elif difficult > 0 and score > 0:
            return 1.0 + 1.0 * difficult / 9
        elif difficult > 0 and score < 0:
            return 1.0 - 0.5 * difficult / 9
        else:
            return 1.0

    raw_scores = token_level_rewards.sum(dim=-1)
    transformed_score = 2 * (raw_scores - 0.5)

    scores = torch.zeros_like(transformed_score)
    for i in range(transformed_score.shape[0]):
        scores[i] = transformed_score[i] * multiplier_scores(difficult_list[i], transformed_score[i])

    id2score = defaultdict(list)
    id2mean, id2std = {}, {}

    bsz = scores.shape[0]
    for i in range(bsz):
        gid = index[i].item()
        id2score[gid].append(scores[i])

    for gid, lst in id2score.items():
        assert len(lst) > 1, "GRPO requires more than 1 sample per group."
        stacked = torch.stack(lst)
        id2mean[gid] = torch.mean(stacked)
        id2std[gid] = torch.std(stacked, unbiased=True)

    for i in range(bsz):
        gid = index[i].item()
        scores[i] = (scores[i] - id2mean[gid]) / (id2std[gid] + eps)

    returns = scores.unsqueeze(-1) * response_mask
    return returns, returns




@torch.no_grad()
def compute_rloo_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        index: `(torch.Tensor)`
            shape: (bs,)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2sum = {}
    bsz = scores.shape[0]
    for i in range(bsz):
        id2score[index[i]].append(scores[i])

    for idx in id2score:
        id2sum[idx] = torch.sum(torch.tensor(id2score[idx]))

    for i in range(bsz):
        sample_num = len(id2score[index[i]])
        assert sample_num > 1, "RLOO needs rollout.n > 1."
        baseline = (id2sum[index[i]] - scores[i]) / (sample_num - 1)
        scores[i] = scores[i] - baseline

    returns = scores.unsqueeze(-1) * response_mask
    return returns, returns


@torch.no_grad()
def compute_reinforce_plus_plus_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, gamma: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    returns = torch.zeros_like(token_level_rewards)
    running_return = 0
    for t in reversed(range(token_level_rewards.shape[1])):
        running_return = token_level_rewards[:, t] + gamma * running_return
        returns[:, t] = running_return
        # Reset after EOS
        running_return = running_return * response_mask[:, t]

    advantages = VF.masked_whiten(returns, response_mask)
    return advantages, returns


@torch.no_grad()
def compute_remax_outcome_advantage(
    token_level_rewards: torch.Tensor, reward_baselines: torch.Tensor, response_mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505

    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    scores = token_level_rewards.sum(dim=-1) - reward_baselines
    returns = scores.unsqueeze(-1) * response_mask
    return returns, returns


def compute_rewards(
    token_level_scores: torch.Tensor,
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    kl_ratio: float,
) -> torch.Tensor:
    kl = log_probs - ref_log_probs
    return token_level_scores - kl * kl_ratio


def average_loss(
    values: torch.Tensor, mask: torch.Tensor, mode: Literal["token", "seq"], eps: float = 1e-8
) -> torch.Tensor:
    """Average the policy loss.

    Args:
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        mask: `(torch.Tensor)`
            shape: (bs, response_length)
        mode: `(Literal["token", "seq"])`
            "token": average the loss in the whole batch
            "seq": average the loss in each sequence then average the mean of the means
        eps: `(float)`
            epsilon value

    Returns:
        loss: `a scalar torch.Tensor`
    """
    if mode == "token":
        return VF.masked_mean(values, mask, eps=eps)
    elif mode == "seq":
        return ((values * mask).sum(-1) / (mask.sum(-1) + eps)).mean()
    else:
        raise NotImplementedError(f"Unknown mode: {mode}.")


def compute_policy_loss(
    old_log_probs: torch.Tensor,
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_ratio_low: float,
    clip_ratio_high: float,
    clip_ratio_dual: float,
    loss_avg_mode: Literal["token", "seq"],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute the clipped policy objective and related metrics for PPO.

    Adapted from https://github.com/huggingface/trl/blob/v0.15.0/trl/trainer/ppo_trainer.py#L568

    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        clip_ratio_low: (float)
            The lower clip range used in PPO. See https://arxiv.org/abs/1707.06347
        clip_ratio_high: (float)
            The higher clip range used in DAPO. See https://arxiv.org/pdf/2503.14476
        clip_ratio_dual: (float)
            The dual clip range used in Dual-clip PPO. See https://arxiv.org/pdf/1912.09729
        loss_avg_mode: (Literal["token", "seq"])
            "token": average the loss in the whole batch
            "seq": average the loss in each sequence then average the mean of the means

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac_higher: (float)
            a float number indicating the fraction of policy gradient loss being clipped to a higher value
        pg_clipfrac_lower: (float)
            a float number indicating the fraction of policy gradient loss being clipped to a lower value
        ppo_kl: (float)
            a float number indicating the mean KL divergence between the old policy and the new policy
        entropy_loss: (float)
            a float number indicating the mean entropy loss

    """
    negative_approx_kl = log_probs - old_log_probs
    # clamp negative_approx_kl to avoid nan kld
    negative_approx_kl = torch.clamp(negative_approx_kl, -20.0, 20.0)
    ratio = torch.exp(negative_approx_kl)
    # clamp the ratio before exp to avoid nan grad
    # see: https://github.com/pytorch/pytorch/issues/10729
    clipped_ratio = torch.exp(
        torch.clamp(negative_approx_kl, np.log(1.0 - clip_ratio_low), np.log(1.0 + clip_ratio_high))
    )

    # pg metrics
    metrics = {"ppo_kl": -negative_approx_kl}
    # use negative log probs as an estimator of entropy loss
    metrics["entropy_loss"] = average_loss(-log_probs, response_mask, mode=loss_avg_mode)

    pg_loss = -advantages * ratio  # -ratio * A
    pg_loss2 = -advantages * clipped_ratio  # -clip(ratio, 1-clip_low, 1+clip_high) * A
    pg_loss3 = -advantages * clip_ratio_dual  # -clip_dual * A

    clipped_pg_loss_higher = torch.max(pg_loss, pg_loss2)  # clip if pg_loss < pg_loss2
    metrics["pg_clipfrac_higher"] = (pg_loss < pg_loss2).float()
    clipped_pg_loss_lower = torch.min(clipped_pg_loss_higher, pg_loss3)  # clip if pg_loss > pg_loss3 and adv < 0
    final_pg_loss = torch.where(advantages < 0, clipped_pg_loss_lower, clipped_pg_loss_higher)
    metrics["pg_clipfrac_lower"] = (clipped_pg_loss_higher > pg_loss3).float() * (advantages < 0).float()

    final_pg_loss = average_loss(final_pg_loss, response_mask, mode=loss_avg_mode)
    metrics = {k: VF.masked_mean(v, response_mask).detach().item() for k, v in metrics.items()}
    return final_pg_loss, metrics


def compute_value_loss(
    vpreds: torch.Tensor,
    returns: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    cliprange_value: float,
    loss_avg_mode: Literal["token", "seq"],
) -> Tuple[torch.Tensor, float]:
    """Compute the value loss.

    Adapted from https://github.com/huggingface/trl/blob/v0.15.0/trl/trainer/ppo_trainer.py#L556

    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange_value: (float)
            The clip range for value net used in PPO. See https://arxiv.org/abs/1707.06347
        loss_avg_mode: (Literal["token", "seq"])
            "token": average the loss in the whole batch
            "seq": average the loss in each sequence then average the mean of the means

    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped

    """
    vpredclipped = torch.clamp(vpreds, values - cliprange_value, values + cliprange_value)
    vf_loss1 = torch.square(vpreds - returns)
    vf_loss2 = torch.square(vpredclipped - returns)
    clipped_vf_losses = torch.max(vf_loss1, vf_loss2)  # clip if vf_loss1 < vf_loss2
    vf_loss = 0.5 * average_loss(clipped_vf_losses, response_mask, mode=loss_avg_mode)
    vf_clipfrac = VF.masked_mean((vf_loss1 < vf_loss2).float(), response_mask).detach().item()
    return vf_loss, vf_clipfrac


def compute_kl(
    log_probs: torch.FloatTensor,
    ref_log_probs: torch.FloatTensor,
    kl_penalty: Literal["kl", "abs", "mse", "low_var_kl", "full"],
) -> torch.Tensor:
    """Compute KL divergence given log_probs and ref_log_probs.

    Adapted from https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L1150

    Args:
        log_probs: torch.Tensor
        ref_log_probs: torch.Tensor
        kl_penalty: str ("kl", "abs", "mse", "low_var_kl", "full")

    Returns:
        kl_div: torch.Tensor

    """
    log_probs, ref_log_probs = log_probs.float(), ref_log_probs.float()
    if kl_penalty == "kl":
        return log_probs - ref_log_probs

    if kl_penalty == "abs":
        return (log_probs - ref_log_probs).abs()

    if kl_penalty == "mse":
        return 0.5 * (log_probs - ref_log_probs).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # URL http://joschu.net/blog/kl-approx.html
    if kl_penalty == "low_var_kl":
        # For numerical stability
        kl = (ref_log_probs - log_probs).clamp(-20.0, 20.0)
        kld = (kl.exp() - kl - 1).contiguous()
        return torch.clamp(kld, min=-10.0, max=10.0)

    if kl_penalty == "full":
        return F.kl_div(ref_log_probs, log_probs, log_target=True, reduction="none").sum(-1)

    raise NotImplementedError(f"Unknown KL penalty: {kl_penalty}.")
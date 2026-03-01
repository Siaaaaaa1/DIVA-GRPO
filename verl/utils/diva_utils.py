import torch
import os
import random
import re
from PIL import Image, ImageDraw, ImageFont
import io
import fcntl
import json
import datetime  # Added missing import

def calculate_difficulty_changes(updates_log):
    """
    Track difficulty changes and calculate the average score.
    :param updates_log: List containing update logs in the format (uid, new_diff, old_diff, avg_score)
    :return: Dictionary containing statistical information
    """
    stats = {
        'new_diff': {},
        'old_diff': {},
        'avg_score': 0.0
    }

    total_entries = 0
    total_score = 0.0

    for uid, new_diff, old_diff, avg_score in updates_log:
        # Record changes in new_diff and old_diff
        if new_diff != 0:
            stats['new_diff'][new_diff] = stats['new_diff'].get(new_diff, 0) + 1
        if old_diff != 0:
            stats['old_diff'][old_diff] = stats['old_diff'].get(old_diff, 0) + 1

        # Accumulate avg_score for average calculation
        total_score += avg_score
        total_entries += 1

    # Calculate average score
    if total_entries > 0:
        stats['avg_avg_score'] = float(total_score / total_entries)

    return stats

def calculate_new_difficulty(avg_score, old_diff, score_ranges, difficulty_changes, min_diff, max_diff):
    for (min_score, max_score), difficulty_change in zip(score_ranges, difficulty_changes):
        if min_score <= avg_score <= max_score:
            new_diff = old_diff + difficulty_change
            break
    else:
        new_diff = old_diff  # Keep original difficulty if no interval matches

    # Limit difficulty value between min_diff and max_diff
    return max(min_diff, min(new_diff, max_diff))

def multiplier(difficult, advantage, weighted_advantage_k):
    # Use element-wise operations instead of scalar judgment
    k = weighted_advantage_k  # Adjustable sensitivity parameter
    
    # Create initial multiplier with the same shape as advantage
    multiplier = torch.ones_like(advantage)
    
    # Find indices of non-zero elements
    nonzero_mask = (advantage != 0)
    
    # Calculate multiplier for non-zero elements
    sign = torch.where(advantage > 0, 1.0, -1.0)
    multiplier[nonzero_mask] = torch.exp(k * difficult * sign[nonzero_mask])
    
    return multiplier

def weighted_advantage(difficult, advantage, weighted_advantage_k):
    return advantage * multiplier(difficult, advantage, weighted_advantage_k)

def log_difficulty_update(id_, old_diff, new_diff, math_reward, path):
    """Log difficulty update."""
    log_entry = {
        "id": id_,
        "old_difficulty": old_diff,
        "new_difficulty": new_diff,
        "math_reward": math_reward,
        "timestamp": datetime.datetime.now().isoformat()
    }
    try:
        with open(path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        print(f"Error logging difficulty update: {e}")

def append_to_json_log(filename, data):
    """Improved JSON log appending method using JSONL format (one JSON object per line)."""
    try:
        with open(filename, "a+") as f:
            # Acquire file lock
            fcntl.flock(f, fcntl.LOCK_EX)
            # Move to end of file
            f.seek(0, 2)
            # Write data (JSONL format does not need commas or brackets)
            json.dump(data, f)
            f.write("\n")
            # Release lock before closing the file
            fcntl.flock(f, fcntl.LOCK_UN)
    except Exception as e:
        print(f"Error writing to {filename}: {str(e)}")

def normalize_advantages(advantages):
    """
    Normalize advantages (Standardization).
    :param advantages: PyTorch Tensor
    :return: Normalized Tensor
    """
    # Calculate mean and standard deviation
    mean = advantages.mean()
    std = advantages.std()
    # Avoid division by zero
    std = std if std > 0 else 1.0
    # Normalize
    normalized = (advantages - mean) / std
    return normalized

def minmax_normalize_advantages(advantages):
    """
    Normalize advantages using Min-Max scaling.
    :param advantages: PyTorch Tensor
    :return: Normalized Tensor
    """
    # Use PyTorch method
    max_abs = torch.max(torch.abs(advantages))
    # If all advantages are 0, return the original array (avoid division by zero)
    if max_abs == 0:
        return advantages
    # Scale to [-1, 1] interval
    normalized = advantages / max_abs
    return normalized

def rms_normalize_advantages(advantages, epsilon=1e-6):
    # Calculate Root Mean Square (RMS) for each sample
    rms = torch.sqrt(torch.mean(torch.square(advantages), dim=-1, keepdim=True) + epsilon)
    # Normalize
    normalized = advantages / rms
    return normalized

def adjust_low_reward_advantages(
    global_advantages: torch.Tensor,
    global_index: list,
    token_level_rewards: torch.Tensor,
    threshold: float = 0.15,
    scale_factor: float = 0.1
) -> torch.Tensor:
    """
    Adjust advantages for global_ids with low rewards.
    
    Args:
        global_advantages: Original calculated global advantages tensor
        global_index: List of global_ids corresponding to global_advantages
        token_level_rewards: Reward tensor for each token
        threshold: Threshold to determine low reward (default 0.15)
        scale_factor: Scaling factor for low reward advantages (default 0.1)
    
    Returns:
        Adjusted global_advantages tensor
    """
    # Create dictionary to record total rewards for each global_id
    global_id_to_rewards = {}
    for gid, rewards in zip(global_index, token_level_rewards):
        if gid not in global_id_to_rewards:
            global_id_to_rewards[gid] = []
        global_id_to_rewards[gid].append(rewards.sum().item())
    
    # Find all global_ids where all rewards are below the threshold
    low_reward_global_ids = {
        gid for gid, rewards in global_id_to_rewards.items()
        if all(r < threshold for r in rewards)
    }
    
    # Clone original advantages to avoid in-place modification
    adjusted_advantages = global_advantages.clone()
    
    # Adjust advantages for low reward global_ids
    for i, gid in enumerate(global_index):
        if gid in low_reward_global_ids:
            adjusted_advantages[i] = adjusted_advantages[i] * scale_factor
    
    return adjusted_advantages

def save_full_vectors_to_json(data, vector_data, output_path):
    """
    Save full vectors to a JSON file (including all related vectors and raw data).
    
    Args:
        data: DataProto containing sample data
        vector_data: Dictionary containing all vector data and raw data to be saved
    """
    existing_lines = []
    
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            existing_lines = [line.strip() for line in f if line.strip()]

    # Prepare data (save processed vectors)
    jsonl_data = [
        json.dumps({
            "id": str(data.non_tensor_batch["id"][i]),
            "local_idx": str(data.non_tensor_batch["uid"][i]),
            "global_idx": str(data.non_tensor_batch["global_uid"][i]),
            "problem": str(data.non_tensor_batch["problem"][i]),
            "index": i,
            "category": data.non_tensor_batch["category"][i],
            "difficulty": data.non_tensor_batch["difficulty"][i],
            # Check if `save_origin_global_advantages` is empty and handle accordingly
            "local_advantages": 
                None if len(vector_data["local_advantages"]) == 0 else (
                    vector_data["local_advantages"][i].tolist()
                    if isinstance(vector_data["local_advantages"][i], torch.Tensor) 
                    else float(vector_data["local_advantages"][i])
                ) if i < len(vector_data["local_advantages"]) else None,
            
            "global_advantages": 
                None if len(vector_data["global_advantages"]) == 0 else (
                    vector_data["global_advantages"][i].tolist()
                    if isinstance(vector_data["global_advantages"][i], torch.Tensor) 
                    else float(vector_data["global_advantages"][i])
                ) if i < len(vector_data["global_advantages"]) else None,

            "advantages": 
                None if len(vector_data["advantages"]) == 0 else (
                    vector_data["advantages"][i].tolist()
                    if isinstance(vector_data["advantages"][i], torch.Tensor) 
                    else float(vector_data["advantages"][i])
                ) if i < len(vector_data["advantages"]) else None,

            "token_rewards_score": 
                None if len(vector_data["token_rewards"]) == 0 else (
                    vector_data["token_rewards"][i].tolist()
                    if isinstance(vector_data["token_rewards"][i], torch.Tensor) 
                    else float(vector_data["token_rewards"][i])
                ) if i < len(vector_data["token_rewards"]) else None,

            "save_origin_global_advantages": 
                None if len(vector_data["save_origin_global_advantages"]) == 0 else (
                    vector_data["save_origin_global_advantages"][i].tolist()[:5]
                    if isinstance(vector_data["save_origin_global_advantages"][i], torch.Tensor) 
                    else vector_data["save_origin_global_advantages"][i][:5]
                ) if i < len(vector_data["save_origin_global_advantages"]) else None,
            
            "save_origin_local_advantages": 
                None if len(vector_data["save_origin_local_advantages"]) == 0 else (
                    vector_data["save_origin_local_advantages"][i].tolist()[:5]
                    if isinstance(vector_data["save_origin_local_advantages"][i], torch.Tensor) 
                    else vector_data["save_origin_local_advantages"][i][:5]
                ) if i < len(vector_data["save_origin_local_advantages"]) else None,

            "save_WBN_global_advantages": 
                None if len(vector_data["save_WBN_global_advantages"]) == 0 else (
                    vector_data["save_WBN_global_advantages"][i].tolist()[:5]
                    if isinstance(vector_data["save_WBN_global_advantages"][i], torch.Tensor) 
                    else vector_data["save_WBN_global_advantages"][i][:5]
                ) if i < len(vector_data["save_WBN_global_advantages"]) else None,

            "save_NORM_global_advantages": 
                None if len(vector_data["save_NORM_global_advantages"]) == 0 else (
                    vector_data["save_NORM_global_advantages"][i].tolist()[:5]
                    if isinstance(vector_data["save_NORM_global_advantages"][i], torch.Tensor) 
                    else vector_data["save_NORM_global_advantages"][i][:5]
                ) if i < len(vector_data["save_NORM_global_advantages"]) else None,

            "save_NORM_local_advantages": 
                None if len(vector_data["save_NORM_local_advantages"]) == 0 else (
                    vector_data["save_NORM_local_advantages"][i].tolist()[:5]
                    if isinstance(vector_data["save_NORM_local_advantages"][i], torch.Tensor) 
                    else vector_data["save_NORM_local_advantages"][i][:5]
                ) if i < len(vector_data["save_NORM_local_advantages"]) else None,

            "save_WAN_global_advantages": 
                None if len(vector_data["save_WAN_global_advantages"]) == 0 else (
                    vector_data["save_WAN_global_advantages"][i].tolist()[:5]
                    if isinstance(vector_data["save_WAN_global_advantages"][i], torch.Tensor) 
                    else vector_data["save_WAN_global_advantages"][i][:5]
                ) if i < len(vector_data["save_WAN_global_advantages"]) else None,

            "save_RRB_global_advantages": 
                None if len(vector_data["save_RRB_global_advantages"]) == 0 else (
                    vector_data["save_RRB_global_advantages"][i].tolist()[:5]
                    if isinstance(vector_data["save_RRB_global_advantages"][i], torch.Tensor) 
                    else vector_data["save_RRB_global_advantages"][i][:5]
                ) if i < len(vector_data["save_RRB_global_advantages"]) else None,

            "save_RRB_local_advantages": 
                None if len(vector_data["save_RRB_local_advantages"]) == 0 else (
                    vector_data["save_RRB_local_advantages"][i].tolist()[:5]
                    if isinstance(vector_data["save_RRB_local_advantages"][i], torch.Tensor) 
                    else vector_data["save_RRB_local_advantages"][i][:5]
                ) if i < len(vector_data["save_RRB_local_advantages"]) else None,
        })
        for i in range(len(data.non_tensor_batch["id"]))
    ]

    # Merge data and write to file (append mode)
    with open(output_path, "a", encoding="utf-8") as f:
        for line in jsonl_data:
            f.write(line + "\n")

    print(f"Full vector data saved. New samples: {len(jsonl_data)}, Total samples: {len(existing_lines) + len(jsonl_data)}")


def save_full_vectors_to_json_origin_grpo(data, vector_data, output_path):
    """
    Save full vectors to a JSON file (including all related vectors and raw data).
    
    Args:
        data: DataProto containing sample data
        vector_data: Dictionary containing all vector data and raw data to be saved
    """
    existing_lines = []
    
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            existing_lines = [line.strip() for line in f if line.strip()]

    # Prepare data (save processed vectors)
    jsonl_data = [
        json.dumps({
            "id": str(data.non_tensor_batch["id"][i]),
            "local_idx": str(data.non_tensor_batch["uid"][i]),
            "problem": str(data.non_tensor_batch["problem"][i]),
            "index": i,
            "category": data.non_tensor_batch["category"][i],
            # Check if `save_origin_global_advantages` is empty and handle accordingly
            "advantages": 
                None if len(vector_data["advantages"]) == 0 else (
                    vector_data["advantages"][i].tolist()
                    if isinstance(vector_data["advantages"][i], torch.Tensor) 
                    else float(vector_data["advantages"][i])
                ) if i < len(vector_data["advantages"]) else None,

             "token_rewards_score": 
                None if len(vector_data["token_rewards"]) == 0 else (
                    vector_data["token_rewards"][i].tolist()
                    if isinstance(vector_data["token_rewards"][i], torch.Tensor) 
                    else float(vector_data["token_rewards"][i])
                ) if i < len(vector_data["token_rewards"]) else None
        })
        for i in range(len(data.non_tensor_batch["id"]))
    ]

    # Merge data and write to file (append mode)
    with open(output_path, "a", encoding="utf-8") as f:
        for line in jsonl_data:
            f.write(line + "\n")

    print(f"Full vector data saved. New samples: {len(jsonl_data)}, Total samples: {len(existing_lines) + len(jsonl_data)}")
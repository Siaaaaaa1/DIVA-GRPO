import copy
import logging
import math
import os
import random
from datetime import datetime
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from PIL import Image as ImageModule
from PIL.Image import Image as ImageObject


from ..difficulty_variation.difficulty_utils import (
    rotate_image,
    add_gaussian_noise,
    add_salt_pepper_noise,
    add_speckle_noise,
    add_blur,
    add_low_resolution,
    add_text_to_image_with_space
    # ,get_variant_text
)

# ==================== Constants and Configuration ====================

# Difficulty adjustment parameters
DIFFICULTY_ADJUSTMENT = {
    "low_threshold": -7,
    "medium_threshold": -5,
    "high_threshold": -3
}

# Noise intensity thresholds
NOISE_INTENSITY_THRESHOLDS = {
    "low": 0.25,
    "medium": 0.45
}

# Image transformation function mapping
IMAGE_TRANSFORMS = {
    "rotate": rotate_image,
    "gaussian": add_gaussian_noise,
}

# Difficulty sampling mapping (used for generate_varent_difficulty_samples)
DIFFICULTY_MAPPING = {
    1: [9], 2: [8], 3: [7], 4: [6], 5: [6],
    6: [5], 7: [5], 8: [5], 9: [4]
}

# Difficulty mean (used for normal distribution sampling)
DIFFICULTY_MEAN = {
    1: 8, 2: 7.5, 3: 7, 4: 6.5, 5: 6,
    6: 5.5, 7: 4.5, 8: 3.5, 9: 3
}

# ==================== Basic Utility Functions ====================

def update_item_metadata(
    item: Dict[str, Any],
    difficulty_delta: int,
    category: str
) -> Dict[str, Any]:
    """Uniformly update item metadata"""
    item['difficulty'] += difficulty_delta
    item['category'] = category
    return item

def select_variant_text(item: Dict[str, Any], probability: float = 0.5) -> str:
    """Randomly select variant text or original problem"""
    if random.random() < probability and item.get('variant') is not None:
        return random.choice(item['variant'])
    return item['problem']

# ==================== Image Processing Functions ====================

def gen_only_vision(
    noise_intensity: float,
    difficulty_delta: int,
    origin_item: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate pure vision problem: add text to the image"""
    if random.random() < 0.5:
        origin_item['problem'] = select_variant_text(origin_item)
    
    origin_item['images']['bytes'] = add_text_to_image_with_space(
        origin_item['images']['bytes'],
        origin_item['problem'],
        text_height_ratio=0.22,
        position=None
    )
    origin_item['problem'] = "As shown in the <image>."
    
    return update_item_metadata(
        origin_item,
        difficulty_delta,
        f'only_vision_{noise_intensity}'
    )

def gen_speckle_noise(
    noise_intensity: float,
    difficulty_delta: int,
    origin_item: Dict[str, Any]
) -> Dict[str, Any]:
    """Add speckle noise"""
    origin_item['images']['bytes'] = add_speckle_noise(
        origin_item['images']['bytes'],
        noise_intensity
    )
    return update_item_metadata(
        origin_item,
        difficulty_delta,
        f'speckle_{noise_intensity}'
    )

def gen_gaussian_noise(
    noise_intensity: float,
    difficulty_delta: int,
    origin_item: Dict[str, Any]
) -> Dict[str, Any]:
    """Add Gaussian noise"""
    origin_item['images']['bytes'] = add_gaussian_noise(
        origin_item['images']['bytes'],
        noise_intensity
    )
    origin_item['problem'] = select_variant_text(origin_item)
    return update_item_metadata(
        origin_item,
        difficulty_delta,
        f'gauss_{noise_intensity}'
    )

def gen_salt_pepper_noise(
    noise_intensity: float,
    difficulty_delta: int,
    origin_item: Dict[str, Any]
) -> Dict[str, Any]:
    """Add salt and pepper noise"""
    origin_item['images']['bytes'] = add_salt_pepper_noise(
        origin_item['images']['bytes'],
        noise_intensity
    )
    origin_item['problem'] = select_variant_text(origin_item)
    return update_item_metadata(
        origin_item,
        difficulty_delta,
        f'salt_{noise_intensity}'
    )

def gen_blur(
    noise_intensity: float,
    difficulty_delta: int,
    origin_item: Dict[str, Any]
) -> Dict[str, Any]:
    """Add blur effect"""
    origin_item['images']['bytes'] = add_blur(
        origin_item['images']['bytes'],
        noise_intensity
    )
    return update_item_metadata(
        origin_item,
        difficulty_delta,
        f'blur_{noise_intensity}'
    )

def gen_low_resolution(
    noise_intensity: float,
    difficulty_delta: int,
    origin_item: Dict[str, Any]
) -> Dict[str, Any]:
    """Reduce image resolution"""
    origin_item['images']['bytes'] = add_low_resolution(
        origin_item['images']['bytes'],
        noise_intensity
    )
    return update_item_metadata(
        origin_item,
        difficulty_delta,
        f'low_resolution_{noise_intensity}'
    )

def gen_rotate(
    noise_intensity: float,
    difficulty_delta: int,
    origin_item: Dict[str, Any]
) -> Dict[str, Any]:
    """Rotate image (angle dynamically calculated based on noise intensity)"""
    # Determine rotation step size based on noise intensity
    if noise_intensity < NOISE_INTENSITY_THRESHOLDS["low"]:
        angle_step = 45
    elif noise_intensity < NOISE_INTENSITY_THRESHOLDS["medium"]:
        angle_step = 30
    else:
        angle_step = 1

    # Calculate possible rotation angles (angles between 1-359 degrees divisible by angle_step)
    max_multiple = (359) // angle_step
    chosen_multiple = random.randint(1, max(1, max_multiple))
    rotation_angle = chosen_multiple * angle_step

    # Apply rotation
    origin_item['images']['bytes'] = rotate_image(
        origin_item['images']['bytes'],
        rotation_angle
    )
    
    # Update problem description
    variant_text = select_variant_text(origin_item)
    origin_item['problem'] = (
        f"{variant_text} This image has been rotated by {rotation_angle} degrees. "
        "Please mentally rotate it back and solve the problem."
    )
    
    return update_item_metadata(
        origin_item,
        difficulty_delta,
        f'rotate_{angle_step}'
    )

# ==================== Text and Problem Processing Functions ====================

def gen_variant_text(
    difficulty_delta: int,
    origin_item: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate text variant problem (does not change difficulty)"""
    origin_item['problem'] = select_variant_text(origin_item)
    origin_item['category'] = 'varient_text'
    return origin_item

def gen_origin_sample(
    difficulty_delta: int,
    origin_item: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate original sample (does not change difficulty)"""
    origin_item['problem'] = origin_item['problem']
    origin_item['category'] = 'origin_item'
    return origin_item

def gen_ground_truth(
    difficulty_delta: int,
    origin_item: Dict[str, Any]
) -> Dict[str, Any]:
    """Provide the correct answer directly in the problem"""
    origin_item['problem'] = (
        f"{origin_item['problem']}\n"
        f"The correct answer to this question is: {origin_item['answer']}. "
        "Please provide detailed reasoning and arrive at the final result."
    )
    return update_item_metadata(
        origin_item,
        difficulty_delta,
        'ground_truth'
    )

def gen_think_step(
    difficulty_delta: int,
    origin_item: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Provide thinking step guidance based on difficulty level
    
    Args:
        difficulty_delta: Negative value, the smaller the value, the more steps provided
        origin_item: Dictionary containing original problem and step2 field
    """
    if not origin_item.get('step2'):
        origin_item['category'] = 'origin_problem_think_step'
        return origin_item

    # Determine number of steps based on difficulty adjustment value
    total_steps = len(origin_item['step2'])
    
    if difficulty_delta <= DIFFICULTY_ADJUSTMENT["low_threshold"]:
        num_steps = total_steps
    elif difficulty_delta <= DIFFICULTY_ADJUSTMENT["medium_threshold"]:
        num_steps = min(3, total_steps)
    elif difficulty_delta <= DIFFICULTY_ADJUSTMENT["high_threshold"]:
        num_steps = min(2, total_steps)
    else:
        num_steps = 1

    # Extract and concatenate thinking steps
    thinking_steps = " ".join(origin_item['step2'][:num_steps])
    
    # Enhance problem description
    origin_item['problem'] = (
        f"{origin_item['problem']} I will now provide some thinking prompts. "
        "Please output the complete thought process and answer from the beginning, "
        f"without skipping any steps. : {thinking_steps}."
    )
    
    return update_item_metadata(
        origin_item,
        difficulty_delta,
        f'guided_thinking_{num_steps}'
    )

# ==================== Composite Transformation Functions ====================

def apply_random_transformations(
    num_transforms: int,
    noise_intensity: float,
    difficulty_delta: int,
    origin_item: Dict[str, Any]
) -> Dict[str, Any]:
    """Randomly apply multiple image transformations"""
    # For mixed text-image content, only rotate
    if origin_item.get('text_in_image', False):
        selected_keys = ["rotate"]
        num_transforms = 1
    else:
        selected_keys = random.sample(
            list(IMAGE_TRANSFORMS.keys()),
            min(num_transforms, len(IMAGE_TRANSFORMS))
        )

    # Randomly replace problem with variant text
    variant_threshold = 0.2 + 0.07 * noise_intensity
    if random.random() < variant_threshold:
        origin_item['problem'] = select_variant_text(origin_item)

    # Apply selected transformations
    for key in selected_keys:
        if key == "rotate":
            origin_item['problem'] = (
                f"{origin_item['problem']} "
                "This image has been rotated. Please mentally rotate it back and solve the problem."
            )
        
        origin_item['images']['bytes'] = IMAGE_TRANSFORMS[key](
            origin_item['images']['bytes'],
            noise_intensity
        )

    # Generate category name
    category_name = f"random_{num_transforms}_{noise_intensity}_" + "_".join(selected_keys)
    
    return update_item_metadata(origin_item, difficulty_delta, category_name)

# ==================== Difficulty Sampling Functions ====================

def get_difficulty_sample(base_difficulty: int) -> Optional[List[int]]:
    """Get deterministic difficulty sample based on base difficulty"""
    return DIFFICULTY_MAPPING.get(base_difficulty)

def generate_variant_difficulty_samples(
    base_difficulty: int,
    num_samples: int = 2,
    sigma: float = 1.0
) -> List[int]:
    """
    Generate difficulty samples
    
    Args:
        base_difficulty: Base difficulty value (1-9)
        num_samples: Number of samples
        sigma: Standard deviation for normal distribution
    """
    if num_samples == 1:
        sample = get_difficulty_sample(base_difficulty)
        return sample if sample else []
    else:
        # Generate samples using normal distribution
        mean = DIFFICULTY_MEAN.get(base_difficulty, 5)
        samples = np.random.normal(mean, sigma, num_samples)
        samples = np.round(samples).astype(int)
        samples = np.clip(samples, 1, 9)
        return samples.tolist()

# ==================== Main Generation Function ====================

# Difficulty function map
DIFFICULTY_FUNCTION_MAP = {
    -8: [partial(gen_think_step, -8)],
    -7: [partial(gen_think_step, -7)],
    -6: [partial(gen_think_step, -6)],
    -5: [partial(gen_think_step, -5)],
    -4: [partial(gen_think_step, -4)],
    -3: [partial(gen_think_step, -3)],
    -2: [partial(gen_think_step, -2)],
    -1: [partial(gen_think_step, -1)],
    0: [partial(gen_variant_text, 0)],
    1: [
        partial(gen_origin_sample, 1),
        partial(gen_origin_sample, 1),
        partial(gen_variant_text, 1),
        partial(gen_gaussian_noise, 0.30, 1),
        partial(gen_rotate, 0.30, 1),
        partial(gen_variant_text, 1)
    ],
    2: [
        partial(gen_gaussian_noise, 0.35, 2),
        partial(gen_rotate, 0.35, 2),
        partial(gen_variant_text, 2),
        partial(gen_only_vision,0.50, 2),
        partial(gen_variant_text, 2)
    ],
    3: [
        partial(gen_gaussian_noise, 0.40, 3),
        partial(gen_rotate, 0.40, 3),
        partial(gen_variant_text, 3),
        partial(gen_only_vision,0.40, 3)
    ],
    4: [partial(gen_gaussian_noise, 0.45, 3), partial(gen_rotate, 0.45, 3), partial(apply_random_transformations, 2, 0.30, 4),partial(apply_random_transformations, 2, 0.30, 4),partial(gen_only_vision,0.50, 2)],
    5: [partial(apply_random_transformations, 2, 0.30, 4),partial(apply_random_transformations, 2, 0.35, 5),partial(gen_only_vision,0.50, 2)],
    6: [partial(apply_random_transformations, 2, 0.30, 4),partial(apply_random_transformations, 2, 0.40, 6),partial(gen_only_vision,0.50, 2)],
    7: [partial(apply_random_transformations, 2, 0.30, 4),partial(apply_random_transformations, 2, 0.45, 7),partial(gen_only_vision,0.50, 2)],
    8: [partial(apply_random_transformations, 2, 0.30, 4),partial(apply_random_transformations, 2, 0.50, 8),partial(gen_only_vision,0.50, 2)],
}

def generate_variants(
    origin_item: Dict[str, Any],
    variant_num: int,
    allow_multiple_thinking: bool = False
) -> List[Dict[str, Any]]:
    """
    Generate a specified number of item variants
    
    Args:
        origin_item: Original item
        variant_num: Number of variants
        allow_multiple_thinking: Whether to allow multiple thinking step variants
    """
    # Generate difficulty samples
    difficulty_samples = generate_variant_difficulty_samples(
        origin_item['difficulty'],
        variant_num
    )
    
    # Calculate difficulty difference
    difficulty_deltas = [d - origin_item['difficulty'] for d in difficulty_samples]
    
    # Handle thinking step variants (only one allowed)
    if not allow_multiple_thinking:
        negative_found = False
        processed_deltas = []
        for delta in difficulty_deltas:
            if delta < 0:
                if not negative_found:
                    processed_deltas.append(delta)
                    negative_found = True
                else:
                    # Replace with random positive value
                    processed_deltas.append(random.choice([0, 1, 2, 3]))
            else:
                processed_deltas.append(delta)
        difficulty_deltas = processed_deltas
    
    # Generate variants
    variant_list = [origin_item]
    
    for idx, delta in enumerate(difficulty_deltas):
        new_item = copy.deepcopy(origin_item)
        
        # Get and execute processing function for corresponding difficulty
        func_list = DIFFICULTY_FUNCTION_MAP.get(delta, [])
        
        if func_list:
            selected_func = random.choice(func_list)
            
            try:
                variant_question = selected_func(new_item)
                variant_list.append(variant_question)
            except Exception as e:
                # Optionally log specific error or re-raise
                raise
    
    return variant_list
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

import math
import os
from collections import defaultdict
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union
from tqdm import tqdm
import numpy as np
import torch
from datasets import load_dataset
from datasets import Dataset as ds_Dataset
from jinja2 import Template
from PIL import Image
from PIL.Image import Image as ImageObject
from qwen_vl_utils.vision_process import fetch_video
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin
import multiprocessing
from multiprocessing import Pool
from functools import partial
from ..models.transformers.qwen2_vl import get_rope_index
from . import torch_functional as VF
from ..difficulty_variation.difficulty_utils import rotate_image, add_gaussian_noise, add_salt_pepper_noise, add_speckle_noise, add_blur, add_low_resolution
import random
import json
import ast
import base64
import binascii
import copy
from pprint import pprint
import re

def format_dict(d, indent=0, max_value_length=80, skip_keys=None):
    if skip_keys is None:
        skip_keys = {'images'}
    indent_str = ' ' * indent
    result = []
    for key, value in d.items():
        if key in skip_keys:
            if isinstance(value, (list, tuple)):
                content = f"<{len(value)} items> [{type(value[0]).__name__}...]" if value else "<empty>"
            elif isinstance(value, dict):
                content = f"<dict with {len(value)} keys>"
            else:
                content = f"<{type(value).__name__}>"
            result.append(f"{indent_str}{key}: {content}")
        else:
            if isinstance(value, dict):
                formatted = format_dict(value, indent + 4, max_value_length, skip_keys)
                result.append(f"{indent_str}{key}: {'{'}\n{formatted}\n{indent_str}{'}'}")
            elif isinstance(value, (list, tuple)) and value and isinstance(value[0], dict):
                items = []
                for i, item in enumerate(value[:3]):
                    items.append(f"{indent_str}    {i}: {'{'}\n{format_dict(item, indent + 8, max_value_length, skip_keys)}\n{indent_str}    {'}'}")
                if len(value) > 3:
                    items.append(f"{indent_str}    ...({len(value)-3} more items)")
                result.append(f"{indent_str}{key}: [\n" + "\n".join(items) + f"\n{indent_str}]")
            else:
                str_value = str(value)
                if len(str_value) > max_value_length:
                    str_value = str_value[:max_value_length] + "..."
                result.append(f"{indent_str}{key}: {str_value}")
    return " ".join(result)


def get_structure(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: get_structure(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [get_structure(obj[0])] if obj else []
    else:
        return type(obj).__name__

def collate_fn(features: List[Union[Dict[str, Any], List[Dict[str, Any]]]]) -> Dict[str, Any]:
    if isinstance(features[0], list):
        features = features[0]
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)
    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)
    for key, value in non_tensors.items():
        non_tensors[key] = np.array(value, dtype=object)
    return {**tensors, **non_tensors}

from typing import List, Dict, Any
import torch
import numpy as np
from collections import defaultdict

def collate_fn_DA(features: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
    flat_features = [item for sublist in features for item in sublist]
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in flat_features:
        for key, value in feature.items():
            if torch.is_tensor(value):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)
    for key in list(tensors.keys()):
        try:
            tensors[key] = torch.stack(tensors[key], dim=0)
        except RuntimeError:
            tensors[key] = torch.cat(tensors[key], dim=0)
            print(f"Warning: field {key} used torch.cat instead of stack")
    for key in non_tensors:
        try:
            non_tensors[key] = np.array(non_tensors[key], dtype=object)
        except Exception as e:
            print(f"Failed to convert field {key} to numpy array: {str(e)}")
            non_tensors[key] = non_tensors[key]
    return {**tensors, **non_tensors}

normal_mean_dict = {
    1:8,
    2:7.5,
    3:7,
    4:6.5,
    5:6,
    6:5,
    7:4,
    8:3.5,
    9:3,
}

normal_mean_dict_more_think = {
    1:8,
    2:7.5,
    3:7,
    4:6.5,
    5:6,
    6:5,
    7:4,
    8:3,
    9:2,
}

def generate_varent_difficulty_samples_2(x, n=2, sigma=1.0):
    if n < 2:
        raise ValueError("n must be ≥ 2 to ensure samples on both sides of the mean")
    mu = normal_mean_dict_more_think[x]
    samples = np.random.normal(mu, sigma, n)
    samples = np.round(samples).astype(int)
    samples = np.clip(samples, 1, 9)
    return samples

normal_mean_dict_up_sampling = {
    1:9,
    2:8.75,
    3:8.5,
    4:8,
    5:7.75,
    6:7.5,
    7:7.25,
    8:7,
    9:7,
}

from PIL import Image, ImageDraw, ImageFont
import io

from PIL import Image, ImageDraw, ImageFont
import io

def _load_font(font_path, size):
    tried = []
    if font_path and os.path.exists(font_path):
        try:
            return ImageFont.truetype(font_path, size)
        except Exception as e:
            tried.append((font_path, str(e)))
    for fp in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
               "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"]:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception as e:
                tried.append((fp, str(e)))
    return ImageFont.load_default()

def _measure_text_width(draw, text, font):
    bbox = draw.textbbox((0,0), text, font=font)
    return bbox[2] - bbox[0]

def _line_height(font):
    try:
        ascent, descent = font.getmetrics()
        return ascent + descent
    except Exception:
        img = Image.new("L", (10,10), 0)
        dr = ImageDraw.Draw(img)
        bbox = dr.textbbox((0,0), "Hg", font=font)
        return bbox[3] - bbox[1]

def _wrap_text_cjk_safe(draw, text, font, max_width):
    lines = []
    for paragraph in str(text).split("\n"):
        tokens = re.findall(r'\s+|[^\s\u4e00-\u9fff]+|[\u4e00-\u9fff]', paragraph)
        line = ""
        for token in tokens:
            candidate = line + token
            if _measure_text_width(draw, candidate, font) <= max_width:
                line = candidate
            else:
                if line:
                    lines.append(line.rstrip())
                if _measure_text_width(draw, token, font) > max_width:
                    sub_line = ""
                    for ch in token:
                        candidate2 = sub_line + ch
                        if _measure_text_width(draw, candidate2, font) <= max_width:
                            sub_line = candidate2
                        else:
                            if sub_line:
                                lines.append(sub_line)
                            sub_line = ch
                    if sub_line:
                        line = sub_line
                    else:
                        line = ""
                else:
                    line = token
        if line:
            lines.append(line.rstrip())
    return lines

def _layout_lines(draw, text, font, max_width, space_height, padding, line_spacing_ratio):
    lines = _wrap_text_cjk_safe(draw, text, font, max_width)
    lh = _line_height(font)
    line_spacing = max(1, int(lh * line_spacing_ratio))
    total_text_height = lh * len(lines) + line_spacing * (len(lines) - 1) if lines else 0
    fits = total_text_height <= max(0, space_height - 2 * padding)
    return fits, lines, lh, line_spacing, total_text_height

def add_text_to_image_with_space(
    image_byte,
    text,
    text_height_ratio=0.2,
    position=None,
    font_path="/mmu_cd_ssd/zhangzhenyu06/workspace/fonts/arimo.ttf",
    max_font_size=100,
    min_font_size=12,
    padding=12,
    line_spacing_ratio=0.2,
    allow_expand=True,
    horizontal_margin=12
):
    try:
        image = Image.open(io.BytesIO(image_byte))
    except Exception as e:
        raise ValueError(f"Failed to parse image data: {e}")
    if image.mode != "RGB":
        image = image.convert("RGB")
    img_width, img_height = image.size
    space_height = max(1, int(img_height * float(text_height_ratio)))
    if position not in ("top", "bottom"):
        position = random.choice(["top", "bottom"])
    total_height = img_height + space_height
    canvas = Image.new("RGB", (img_width, total_height), (255, 255, 255))
    text_y_offset = 0 if position == "top" else img_height
    if position == "top":
        canvas.paste(image, (0, space_height))
    else:
        canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    lo, hi = min_font_size, max_font_size
    best = None
    max_line_width = max(1, img_width - 2 * (padding + horizontal_margin))
    while lo <= hi:
        mid = (lo + hi) // 2
        font = _load_font(font_path, mid)
        fits, lines, lh, line_spacing, total_text_h = _layout_lines(
            draw, text, font, max_line_width, space_height, padding, line_spacing_ratio
        )
        if fits:
            best = (mid, font, lines, lh, line_spacing, total_text_h)
            lo = mid + 1
        else:
            hi = mid - 1
    if best is None:
        font = _load_font(font_path, min_font_size)
        fits, lines, lh, line_spacing, total_text_h = _layout_lines(
            draw, text, font, max_line_width, space_height, padding, line_spacing_ratio
        )
        if not fits and allow_expand:
            needed_space = total_text_h + 2 * padding
            new_total_height = img_height + needed_space
            new_canvas = Image.new("RGB", (img_width, new_total_height), (255, 255, 255))
            if position == "top":
                new_canvas.paste(image, (0, needed_space))
                text_y_offset = 0
            else:
                new_canvas.paste(image, (0, 0))
                text_y_offset = img_height
            canvas = new_canvas
            draw = ImageDraw.Draw(canvas)
            space_height = needed_space
        chosen = (min_font_size, font, lines, lh, line_spacing, total_text_h)
    else:
        chosen = best
    font_size, font, lines, lh, line_spacing, total_text_h = chosen
    fits, lines, lh, line_spacing, total_text_h = _layout_lines(
        draw, text, font, max_line_width, space_height, padding, line_spacing_ratio
    )
    start_y = text_y_offset + max(padding, (space_height - total_text_h) // 2)
    y = start_y
    for i, line in enumerate(lines):
        line_width = _measure_text_width(draw, line, font)
        x = max(padding, (img_width - line_width) // 2)
        draw.text((x, y), line, font=font, fill="black")
        y += lh + line_spacing
    out = io.BytesIO()
    canvas.save(out, format="PNG")
    out.seek(0)
    return out.getvalue()


def gen_only_vision(noise_intensity, diff, origin_item):
    if random.random() < 0.5:
         origin_item['problem'] = random.choices(origin_item['variant'])[0] if origin_item['variant'] is not None else origin_item['problem']
    origin_item['images']['bytes'] = add_text_to_image_with_space(origin_item['images']['bytes'], origin_item['problem'], text_height_ratio=0.22, position=None)
    origin_item['difficulty'] = origin_item['difficulty'] + diff
    origin_item['category'] = 'only_vision_'+str(noise_intensity)
    origin_item['problem'] = "As shown in the <image>."
    return origin_item


def generate_varent_difficulty_samples(x, n=2, sigma=1.0):
    if n < 2:
        raise ValueError("n must be ≥ 2 to ensure samples on both sides of the mean")
    mu = normal_mean_dict[x]
    samples = np.random.normal(mu, sigma, n)
    samples = np.round(samples).astype(int)
    samples = np.clip(samples, 1, 9)
    return samples



def generate_varent_difficulty_samples_up_sampling(x, n=2, sigma=1.0):
    if n < 2:
        raise ValueError("n must be ≥ 2 to ensure samples on both sides of the mean")
    mu = normal_mean_dict_up_sampling[x]
    samples = np.random.normal(mu, sigma, n)
    samples = np.round(samples).astype(int)
    samples = np.clip(samples, 1, 9)
    return samples


import numpy as np

def gen_speckle_noise(noise_intensity, diff, origin_item):
    origin_item['images']['bytes'] = add_speckle_noise(origin_item['images']['bytes'], noise_intensity)
    origin_item['difficulty'] = origin_item['difficulty'] + diff
    origin_item['category'] = 'speckle_' + str(noise_intensity)
    return origin_item

import random

def apply_random_transformations(Num, noise_intensity, diff, origin_item):
    transformation_functions = {
        "rotate": rotate_image,
        "gaussian": add_gaussian_noise,
        "salt": add_salt_pepper_noise,
        "blur": add_blur,
    }
    if 'text_in_image' in origin_item and origin_item['text_in_image'] == True:
        selected_keys = ["rotate"]
        Num = 1
    else:
        selected_keys = random.sample(list(transformation_functions.keys()), Num)
    if random.random() < 0.2 + 0.07 * noise_intensity:
         origin_item['problem'] = random.choices(origin_item['variant'])[0] if origin_item['variant'] is not None else origin_item['problem']
    for key in selected_keys:
        if key == "rotate":
            origin_item['problem'] = origin_item['problem'] + f"This image has been rotated. Please mentally rotate it back and solve the problem."
        origin_item['images']['bytes'] = transformation_functions[key](origin_item['images']['bytes'], noise_intensity)
    origin_item['difficulty'] = origin_item['difficulty'] + diff
    category_name = "random_"+f"{Num}_"+f"{noise_intensity}_"+"_".join(selected_keys)
    origin_item['category'] = category_name
    return origin_item

def gen_think_step(difficulty_adjustment, original_item):
    if 'step2' in original_item and original_item['step2'] is not None:
        if difficulty_adjustment <= -7:
            noise_level = len(original_item['step2'])
            thinking_steps = " ".join(original_item['step2'])
        elif difficulty_adjustment <= -5:
            noise_level = 3
            thinking_steps = " ".join(original_item['step2'][:3])
        elif difficulty_adjustment <= -3:
            noise_level = 2
            thinking_steps = " ".join(original_item['step2'][:2])
        else:
            noise_level = 1
            thinking_steps = " ".join(original_item['step2'][:1])
        original_item['problem'] = (f"{original_item['problem']} I will now provide some thinking prompts. Please output the complete thought process and answer from the beginning, without skipping any steps. : {thinking_steps}.")
        original_item['difficulty'] += difficulty_adjustment
        original_item['category'] = f'guided_thinking_{noise_level}'
    else:
        original_item['category'] = 'origin_problem_think_step'
    return original_item

def gen_blur(noise_intensity, diff, origin_item):
    origin_item['images']['bytes'] = add_blur(origin_item['images']['bytes'], noise_intensity)
    origin_item['difficulty'] = origin_item['difficulty'] + diff
    origin_item['category'] = 'blur_'+str(noise_intensity)
    return origin_item

def gen_low_resolution(noise_intensity, diff, origin_item):
    origin_item['images']['bytes'] = add_low_resolution(origin_item['images']['bytes'], noise_intensity)
    origin_item['difficulty'] = origin_item['difficulty'] + diff
    origin_item['category'] = 'low_resolution_'+str(noise_intensity)
    return origin_item

def gen_origin_item(diff, origin_item):
    origin_item['problem'] = origin_item['problem']
    origin_item['difficulty'] = origin_item['difficulty']
    origin_item['category'] = 'chain_of_though'
    return origin_item

def gen_ground_truth(diff, origin_item):
    origin_item['problem'] = origin_item['problem'] + f"\nThe correct answer to this question is: {origin_item['answer']}. Please provide detailed reasoning and arrive at the final result."
    origin_item['difficulty'] = origin_item['difficulty'] + diff
    origin_item['category'] = 'ground_truth'
    return origin_item

def gen_gauss(noise_intensity, diff, origin_item):
    origin_item['images']['bytes'] = add_gaussian_noise(origin_item['images']['bytes'], noise_intensity)
    origin_item['difficulty'] = origin_item['difficulty'] + diff
    origin_item['category'] = 'gauss_'+str(noise_intensity)
    origin_item['problem'] = get_variant_text(origin_item)
    return origin_item

def gen_origin_question(origin_item):
    origin_item['problem'] = origin_item['problem']
    origin_item['category'] = 'origin_problem'
    return origin_item

def get_variant_text(origin_item):
    if random.random() < 0.5:
        return random.choices(origin_item['variant'])[0] if origin_item['variant'] is not None else origin_item['problem']
    else:
        return origin_item['problem']

def gen_salt(noise_intensity, diff, origin_item):
    origin_item['images']['bytes'] = add_salt_pepper_noise(origin_item['images']['bytes'], noise_intensity)
    origin_item['difficulty'] = origin_item['difficulty'] + diff
    origin_item['category'] = 'salt_'+str(noise_intensity)
    origin_item['problem'] = get_variant_text(origin_item)
    return origin_item

def gen_rotate(noise_intensity, diff, origin_item):
    if noise_intensity<0.25:
        div_angle = 45
    elif noise_intensity<0.45:
        div_angle = 30
    else:
        div_angle = 1
    max_multiple = (360 - 1) // div_angle
    if max_multiple < 1:
        chosen_multiple = 1
    else:
        chosen_multiple = random.randint(1, max_multiple)
    new_angle = chosen_multiple * div_angle
    origin_item['images']['bytes'] = rotate_image(origin_item['images']['bytes'], new_angle)
    origin_item['problem'] = get_variant_text(origin_item) + f"This image has been rotated by {new_angle} degrees. Please mentally rotate it back and solve the problem."
    origin_item['difficulty'] = origin_item['difficulty'] + diff
    origin_item['category'] = 'rotate_'+str(div_angle)
    return origin_item

def gen_var_text(diff, origin_item):
    origin_item['problem'] = random.choices(origin_item['variant'])[0] if origin_item['variant'] is not None else origin_item['problem']
    origin_item['difficulty'] = origin_item['difficulty']
    origin_item['category'] = 'varient_text'
    return origin_item

def smart_convert(s):
    try:
        data = json.loads(s)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, TypeError):
        try:
            data = ast.literal_eval(s)
            if isinstance(data, dict):
                return data
        except (SyntaxError, ValueError, TypeError):
            pass
    try:
        return s.encode('utf-8')
    except (AttributeError, UnicodeEncodeError):
        pass
    if len(s) % 2 == 0 and all(c in "0123456789abcdefABCDEF" for c in s):
        try:
            return bytes.fromhex(s)
        except (ValueError, TypeError):
            pass
    try:
        return base64.b64decode(s)
    except (binascii.Error, TypeError):
        pass
    return s

def process_image(
    image: Union[Dict[str, Any], ImageObject, str], min_pixels: Optional[int], max_pixels: Optional[int]
) -> ImageObject:
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, dict):
        image = Image.open(BytesIO(image["bytes"]))
    elif isinstance(image, bytes):
        image = Image.open(BytesIO(image))
    image.load()
    if max_pixels is not None and (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))
    if min_pixels is not None and (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image

def process_video(
    video: str, min_pixels: Optional[int], max_pixels: Optional[int], video_fps: float, return_fps: bool = False
) -> Union[List[ImageObject], Tuple[List[ImageObject], List[float]]]:
    vision_info = {"video": video, "min_pixels": min_pixels, "max_pixels": max_pixels, "fps": video_fps}
    return fetch_video(vision_info, return_video_sample_fps=return_fps)

difficulty_varient_function_list = {
        -8:[partial(gen_think_step,-8)],
        -7:[partial(gen_think_step,-7)],
        -6:[partial(gen_think_step,-6)],
        -5:[partial(gen_think_step,-5)],
        -4:[partial(gen_think_step,-4)],
        -3:[partial(gen_think_step,-3)],
        -2:[partial(gen_think_step,-2)],
        -1:[partial(gen_think_step,-1)],
        0:[partial(gen_var_text,0)],
        1:[partial(gen_gauss,0.40,1),partial(gen_salt,0.40,1),partial(gen_rotate,0.40,1),partial(gen_blur,0.40,1),partial(gen_low_resolution,0.40,1),partial(gen_var_text,1)],
        2:[partial(gen_gauss,0.50,2),partial(gen_salt,0.50,2),partial(gen_rotate,0.50,2),partial(gen_blur,0.50,2),partial(gen_low_resolution,0.50,2),partial(gen_var_text,2)],
        3:[partial(gen_gauss,0.60,3),partial(gen_salt,0.60,3),partial(gen_rotate,0.60,3),partial(gen_blur,0.60,3),partial(gen_low_resolution,0.60,3),partial(gen_var_text,3)],
        4:[partial(apply_random_transformations,2,0.40,4)],
        5:[partial(apply_random_transformations,2,0.50,5)],
        6:[partial(apply_random_transformations,3,0.40,6)],
        7:[partial(apply_random_transformations,3,0.50,7)],
        8:[partial(apply_random_transformations,4,0.50,8)],
    }

difficulty_varient_function_list_new = {
        -8:[partial(gen_think_step,-8)],
        -7:[partial(gen_think_step,-7)],
        -6:[partial(gen_think_step,-6)],
        -5:[partial(gen_think_step,-5)],
        -4:[partial(gen_think_step,-4)],
        -3:[partial(gen_think_step,-3)],
        -2:[partial(gen_think_step,-2)],
        -1:[partial(gen_think_step,-1)],
        0:[partial(gen_var_text,0)],
        1:[partial(apply_random_transformations,1,0.40,1)],
        2:[partial(apply_random_transformations,1,0.50,2)],
        3:[partial(apply_random_transformations,2,0.40,3)],
        4:[partial(apply_random_transformations,2,0.60,4)],
        5:[partial(apply_random_transformations,3,0.50,5)],
        6:[partial(apply_random_transformations,4,0.50,6)],
        7:[partial(gen_only_vision,0.50,7)],
        8:[partial(gen_only_vision,0.60,8)],
    }

difficulty_varient_function_list_only_text = {
        -8:[partial(gen_think_step,-8)],
        -7:[partial(gen_think_step,-7)],
        -6:[partial(gen_think_step,-6)],
        -5:[partial(gen_think_step,-5)],
        -4:[partial(gen_think_step,-4)],
        -3:[partial(gen_think_step,-3)],
        -2:[partial(gen_think_step,-2)],
        -1:[partial(gen_think_step,-1)],
        0:[partial(gen_var_text,0)],
        1:[partial(apply_random_transformations,1,0.30,1)],
        2:[partial(apply_random_transformations,1,0.45,2)],
        3:[partial(apply_random_transformations,2,0.30,3)],
        4:[partial(apply_random_transformations,2,0.45,4)],
        5:[partial(apply_random_transformations,3,0.30,5)],
        6:[partial(apply_random_transformations,4,0.45,6)],
        7:[partial(gen_only_vision,0.50,7)],
        8:[partial(gen_only_vision,0.60,8)],
    }

no_thinking_sampling_varient_function_list = [
    partial(gen_gauss,0.50,2),
    partial(gen_salt,0.50,2),
    partial(gen_rotate,0.60,2),
    partial(gen_blur,0.50,2),
    partial(gen_low_resolution,0.50,2),
    partial(gen_var_text,2),
    partial(apply_random_transformations,2,0.40,4),
    partial(apply_random_transformations,2,0.50,5),
    partial(apply_random_transformations,3,0.40,6),
    partial(apply_random_transformations,3,0.50,7),
    partial(apply_random_transformations,4,0.50,8)]

def generate_varitent(origin_item, varient_num):
    sample_difficult_list = generate_varent_difficulty_samples(origin_item['difficulty'],varient_num)
    diff_list = sample_difficult_list - origin_item['difficulty']
    varient_list = [origin_item]
    for diff in diff_list:
        new_item = copy.deepcopy(origin_item)
        func_list = difficulty_varient_function_list[diff]
        selected_func = random.choice(func_list)
        varient_question = selected_func(new_item)
        varient_list.append(varient_question)
    return varient_list

import numpy as np

normal_mean_dict_v2 = {
    5: 2,
    4: 3,
    3: 4,
    2: 5,
    1: 6,
}
normal_mean_dict_v3 = {
    9: -5,
    8: -3,
    7: -1,
    6: 1,
    5: 2,
    4: 3,
    3: 4,
    2: 5,
    1: 6,
}

normal_mean_dict_only_text = {
    9: -5,
    8: -3,
    7: -1,
    6: 1,
    5: 2,
    4: 3,
    3: 4,
    2: 5,
    1: 6,
}

def generate_only_text_samples(x, n=2, sigma=1.0):
    if x in normal_mean_dict_only_text:
        mu = normal_mean_dict_only_text[x]
        samples = np.random.normal(mu, sigma, n)
    else:
        raise ValueError(f"Unsupported input x={x}")
    samples = np.round(samples).astype(int)
    samples = np.clip(samples, -8, 8)
    return samples

def generate_more_thinking_samples(x, n=2, sigma=1.0):
    if n < 2 and x not in [6, 7, 8, 9]:
        raise ValueError("n must be ≥ 2 to ensure samples on both sides of the mean")
    if x in normal_mean_dict_v3:
        mu = normal_mean_dict_v3[x]
        samples = np.random.normal(mu, sigma, n)
    else:
        raise ValueError(f"Unsupported input x={x}")
    samples = np.round(samples).astype(int)
    samples = np.clip(samples, -8, 8)
    return samples

def generate_variant_difficulty_samples_v2(x, n=2, sigma=1.0):
    if n < 2:
        if x not in normal_mean_dict_v3:
            raise ValueError(f"When n < 2, x={x} must exist in normal_mean_dict_v2")
        mu = normal_mean_dict_v3[x]
        samples = np.random.normal(mu, sigma, n)
    else:
        if x in normal_mean_dict_v2:
            mu = normal_mean_dict_v2[x]
            samples = np.random.normal(mu, sigma, n)
        elif x == 6:
            samples = np.concatenate([
                np.random.normal(2, sigma, n - 1),
                np.random.normal(-2.5, sigma, 1)
            ])
        elif x == 7:
            samples = np.concatenate([
                np.random.normal(2, sigma, n - 1),
                np.random.normal(-4, sigma, 1)
            ])
        elif x == 8:
            samples = np.concatenate([
                np.random.normal(2, sigma, n - 1),
                np.random.normal(-5.5, sigma, 1)
            ])
        elif x == 9:
            samples = np.concatenate([
                np.random.normal(2, sigma, n - 1),
                np.random.normal(-7, sigma, 1)
            ])
        else:
            raise ValueError(f"Unsupported input x={x}")
    samples = np.round(samples).astype(int)
    samples = np.clip(samples, -8, 8)
    return samples

def generate_varitent_more_thinking(origin_item, varient_num):
    sample_diff = generate_more_thinking_samples(origin_item['difficulty'], varient_num)
    varient_list = [origin_item]
    for diff in sample_diff:
        new_item = copy.deepcopy(origin_item)
        func_list = difficulty_varient_function_list_new[diff]
        selected_func = random.choice(func_list)
        varient_question = selected_func(new_item)
        varient_list.append(varient_question)
    return varient_list

def generate_varitent_gpto3_thinking(origin_item, varient_num):
    sample_diff = generate_variant_difficulty_samples_v2(origin_item['difficulty'], varient_num)
    varient_list = [origin_item]
    for diff in sample_diff:
        new_item = copy.deepcopy(origin_item)
        func_list = difficulty_varient_function_list_new[diff]
        selected_func = random.choice(func_list)
        varient_question = selected_func(new_item)
        varient_list.append(varient_question)
    return varient_list

def generate_varitent_only_text_thinking(origin_item, varient_num):
    sample_diff = generate_only_text_samples(origin_item['difficulty'], varient_num)
    varient_list = [origin_item]
    for diff in sample_diff:
        new_item = copy.deepcopy(origin_item)
        func_list = difficulty_varient_function_list_only_text[diff]
        selected_func = random.choice(func_list)
        varient_question = selected_func(new_item)
        varient_list.append(varient_question)
    return varient_list

def generate_varitent_one_thinking_less_think(origin_item, varient_num):
    sample_difficult_list = generate_varent_difficulty_samples_2(origin_item['difficulty'], varient_num)
    diff_list = sample_difficult_list - origin_item['difficulty']
    negative_found = False
    processed_diff_list = []
    for diff in diff_list:
        if diff < 0:
            if not negative_found:
                processed_diff_list.append(diff)
                negative_found = True
            else:
                processed_diff_list.append(random.choice([1, 2, 3]))
        else:
            processed_diff_list.append(diff)
    varient_list = [origin_item]
    for diff in processed_diff_list:
        new_item = copy.deepcopy(origin_item)
        func_list = difficulty_varient_function_list[diff]
        selected_func = random.choice(func_list)
        varient_question = selected_func(new_item)
        varient_list.append(varient_question)
    return varient_list

def generate_varitent_one_thinking(origin_item, varient_num):
    sample_difficult_list = generate_varent_difficulty_samples(origin_item['difficulty'], varient_num)
    diff_list = sample_difficult_list - origin_item['difficulty']
    negative_found = False
    processed_diff_list = []
    for diff in diff_list:
        if diff < 0:
            if not negative_found:
                processed_diff_list.append(diff)
                negative_found = True
            else:
                processed_diff_list.append(1)
        else:
            processed_diff_list.append(diff)
    varient_list = [origin_item]
    for diff in processed_diff_list:
        new_item = copy.deepcopy(origin_item)
        func_list = difficulty_varient_function_list[diff]
        selected_func = random.choice(func_list)
        varient_question = selected_func(new_item)
        varient_list.append(varient_question)
    return varient_list

def generate_varitent_up_sampling(origin_item, varient_num):
    diff_list = [0] * varient_num
    varient_list = [origin_item]
    for diff in diff_list:
        new_item = copy.deepcopy(origin_item)
        func_list = difficulty_varient_function_list[diff]
        selected_func = random.choice(func_list)
        varient_question = selected_func(new_item)
        varient_list.append(varient_question)
    return varient_list

def generate_varitent_fixed_sampling(origin_item, varient_num):
    origin_item1 = copy.deepcopy(origin_item)
    origin_item2 = copy.deepcopy(origin_item)
    origin_item3 = copy.deepcopy(origin_item)
    varient_list = [origin_item,gen_gauss(0.5,0,origin_item1),gen_salt(0.6,0,origin_item2),gen_rotate(0.5,0,origin_item3)]
    return varient_list

def generate_varitent_random_sampling(origin_item, varient_num):
    sample_difficult_list = generate_varent_difficulty_samples_up_sampling(origin_item['difficulty'],varient_num)
    diff_list = sample_difficult_list - origin_item['difficulty']
    varient_list = [origin_item]
    for diff in diff_list:
        new_item = copy.deepcopy(origin_item)
        selected_func = random.choice(no_thinking_sampling_varient_function_list)
        varient_question = selected_func(new_item)
        varient_list.append(varient_question)
    return varient_list

class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        image_key: str = "images",
        video_key: str = "videos",
        image_dir: Optional[str] = None,
        video_fps: float = 2.0,
        max_prompt_length: int = 1024,
        truncation: str = "error",
        format_prompt: Optional[str] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        filter_overlong_prompts: bool = True,
        filter_overlong_prompts_workers: int = 16,
        Difficulty_Adaptation: Optional[bool] = False,
        Varient_Num: Optional[int] = 0,
        Dataset_Mode: Optional[str] = None
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.video_key = video_key
        self.image_dir = image_dir
        self.video_fps = video_fps
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.Difficulty_Adaptation = Difficulty_Adaptation
        self.Varient_Num = Varient_Num
        self.Dataset_Mode = Dataset_Mode
        if "@" in data_path:
            data_path, data_split = data_path.split("@")
        else:
            data_split = "train"

        if os.path.isdir(data_path):
            file_type = os.path.splitext(os.listdir(data_path)[0])[-1][1:].replace("jsonl", "json")
            self.dataset = load_dataset(file_type, data_dir=data_path, split=data_split)
        elif os.path.isfile(data_path):
            file_type = os.path.splitext(data_path)[-1][1:].replace("jsonl", "json")
            self.dataset = load_dataset(file_type, data_files=data_path, split=data_split)
        else:
            self.dataset = load_dataset(data_path, split=data_split)

        self.format_prompt = None
        if format_prompt:
            with open(format_prompt, encoding="utf-8") as f:
                self.format_prompt = f.read()

        if self.Difficulty_Adaptation == False and filter_overlong_prompts:
            print(f"Before overlong prompts filter datasets length is === {len(self.dataset)}")
            self.dataset = self.dataset.filter(
                self._filter_overlong_prompts,
                desc="Filtering overlong prompts",
                num_proc=filter_overlong_prompts_workers,
            )
            print(f"After overlong prompts filter datasets length is === {len(self.dataset)}")


    def _build_messages(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        prompt_str: str = example[self.prompt_key]
        if self.format_prompt:
            format_prompt = Template(self.format_prompt.strip())
            prompt_str = format_prompt.render(content=prompt_str)

        if self.image_key in example:
            content_list = []
            parts = prompt_str.split("<image>", 1)
            content_list = []
            if parts[0]:
                content_list.append({"type": "text", "text": parts[0]})
            if len(parts) > 1:
                content_list.append({"type": "image"})
                if parts[1]:
                    content_list.append({"type": "text", "text": parts[1]})
            return [{"role": "user", "content": content_list}]
        elif self.video_key in example:
            content_list = []
            for i, content in enumerate(prompt_str.split("<video>")):
                if i != 0:
                    content_list.append({"type": "video"})
                if content:
                    content_list.append({"type": "text", "text": content})
            return [{"role": "user", "content": content_list}]
        else:
            return [{"role": "user", "content": prompt_str}]

    def _filter_overlong_prompts(self, example: Dict[str, Any]) -> bool:
        messages = self._build_messages(example)
        if self.image_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = example[self.image_key]
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):
                images = [os.path.join(self.image_dir, image) for image in images]
            processed_images = [] if len(images) != 0 else None
            if isinstance(images, dict):
                processed_images.append(process_image(images, self.min_pixels, self.max_pixels))
            else:
                for image in images:
                    processed_images.append(process_image(image, self.min_pixels, self.max_pixels))
            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        elif self.video_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            videos = example[self.video_key]
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):
                videos = [os.path.join(self.image_dir, video) for video in videos]
            processed_videos = [] if len(videos) != 0 else None
            for video in videos:
                processed_videos.append(process_video(video, self.min_pixels, self.max_pixels, self.video_fps))
            model_inputs = self.processor(
                videos=processed_videos, text=[prompt], add_special_tokens=False, return_tensors="pt"
            )
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        else:
            input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            return len(input_ids) <= self.max_prompt_length

    def __len__(self):
        return len(self.dataset) 

    def __getitem__(self, index):
        example: dict = self.dataset[index]
        if self.Difficulty_Adaptation == True:
            if self.Dataset_Mode.startswith("more_thinking"):
                varient_list = generate_varitent_more_thinking(example, self.Varient_Num)
                return [self._process_single_example(new_example, is_variant=True) for new_example in varient_list]
            elif self.Dataset_Mode.startswith("gpto3_thinking"):
                varient_list = generate_varitent_gpto3_thinking(example, self.Varient_Num)
                return [self._process_single_example(new_example, is_variant=True) for new_example in varient_list]
            elif self.Dataset_Mode.startswith("only_text_thinking"):
                varient_list = generate_varitent_only_text_thinking(example, self.Varient_Num)
                return [self._process_single_example(new_example, is_variant=True) for new_example in varient_list]
            elif self.Dataset_Mode.startswith("one_thinking"):
                varient_list = generate_varitent_one_thinking(example, self.Varient_Num)
                return [self._process_single_example(new_example, is_variant=True) for new_example in varient_list]
            elif self.Dataset_Mode.startswith("less_thinking"):
                varient_list = generate_varitent_one_thinking_less_think(example, self.Varient_Num)
                return [self._process_single_example(new_example, is_variant=True) for new_example in varient_list]
            elif self.Dataset_Mode.startswith("up_sampling"):
                varient_list = generate_varitent_up_sampling(example, self.Varient_Num)
                return [self._process_single_example(new_example, is_variant=True) for new_example in varient_list]
            elif self.Dataset_Mode.startswith("fixed_sampling"):
                varient_list = generate_varitent_fixed_sampling(example, self.Varient_Num)
                return [self._process_single_example(new_example, is_variant=True) for new_example in varient_list]
            elif self.Dataset_Mode.startswith("random_sampling"):
                varient_list = generate_varitent_random_sampling(example, self.Varient_Num)
                return [self._process_single_example(new_example, is_variant=True) for new_example in varient_list]
            else:
                varient_list = generate_varitent(example, self.Varient_Num)
                return [self._process_single_example(new_example, is_variant=True) for new_example in varient_list]
        else:
            return self._process_single_example(example)

    def _process_single_example(self, example, is_variant=False):
        messages = self._build_messages(example) if not is_variant else self._build_variant_messages(example)
        if self.image_key in example:
            return self._process_image_data(example, messages)
        elif self.video_key in example:
            return self._process_video_data(example, messages)
        else:
            return self._process_text_data(example, messages)

    def _build_variant_messages(self, example):
        prompt_str = example[self.prompt_key]
        if self.format_prompt:
            format_prompt = Template(self.format_prompt.strip())
            prompt_str = format_prompt.render(content=prompt_str)
        if self.image_key in example:
            content_list = []
            parts = prompt_str.split("<image>", 1)
            content_list = []
            if parts[0]:
                content_list.append({"type": "text", "text": parts[0]})
            if len(parts) > 1:
                content_list.append({"type": "image"})
                if parts[1]:
                    content_list.append({"type": "text", "text": parts[1]})
            return [{"role": "user", "content": content_list}]
        elif self.video_key in example:
            content_list = []
            for i, content in enumerate(prompt_str.split("<video>")):
                if i != 0:
                    content_list.append({"type": "video"})
                if content:
                    content_list.append({"type": "text", "text": content})
            return [{"role": "user", "content": content_list}]
        else:
            return [{"role": "user", "content": prompt_str}]

    def _process_image_data(self, example, messages):
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        images = example.pop(self.image_key)
        if self.image_dir is not None and images and isinstance(images[0], str):
            images = [os.path.join(self.image_dir, image) for image in images]
        processed_images = []
        if isinstance(images, dict):
            processed_images.append(process_image(images, self.min_pixels, self.max_pixels))
        else:
            for image in images:
                processed_images.append(process_image(image, self.min_pixels, self.max_pixels))
        model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
        input_ids = model_inputs.pop("input_ids")[0]
        attention_mask = model_inputs.pop("attention_mask")[0]
        example["multi_modal_data"] = {"images": images}
        return self._finalize_example(example, model_inputs, input_ids, attention_mask, prompt)

    def _process_video_data(self, example, messages):
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        videos = example.pop(self.video_key)
        if self.image_dir is not None and videos and isinstance(videos[0], str):
            videos = [os.path.join(self.image_dir, video) for video in videos]
        processed_videos = []
        video_fps_list = []
        for video in videos:
            processed_video, video_fps = process_video(
                video, self.min_pixels, self.max_pixels, self.video_fps, return_fps=True
            )
            processed_videos.append(processed_video)
            video_fps_list.append(video_fps)
        model_inputs = self.processor(
            videos=processed_videos, text=[prompt], add_special_tokens=False, return_tensors="pt"
        )
        if "second_per_grid_ts" in self.processor.model_input_names:
            model_inputs["second_per_grid_ts"] = [2.0 / fps for fps in video_fps_list]
        input_ids = model_inputs.pop("input_ids")[0]
        attention_mask = model_inputs.pop("attention_mask")[0]
        example["multi_modal_data"] = {"videos": videos}
        return self._finalize_example(example, model_inputs, input_ids, attention_mask, prompt)

    def _process_text_data(self, example, messages):
        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
        input_ids = model_inputs.pop("input_ids")[0]
        attention_mask = model_inputs.pop("attention_mask")[0]
        return self._finalize_example(example, model_inputs, input_ids, attention_mask, prompt)

    def update_difficulty(self, updates):
        """
        Batch update difficulty values safely and efficiently.

        Args:
            updates: list of (uid, new_difficulty) tuples

        Returns:
            bool: True if all updates succeeded (no missing ids), else False
        """
        uid_to_diff = {str(uid): new_diff for uid, new_diff in updates}
        print(f"\n[DEBUG] Starting difficulty update for {len(updates)} samples...")

        before_diff_dist = dict(zip(*np.unique(self.dataset['difficulty'], return_counts=True)))

        def apply_update(example):
            uid = str(example.get('id'))
            if uid in uid_to_diff:
                example['difficulty'] = uid_to_diff[uid]
            return example

        self.dataset = self.dataset.map(
            apply_update,
            num_proc=4,
            desc="Updating difficulties"
        )

        all_ids = set(str(x) for x in self.dataset['id'])
        missing_ids = [uid for uid in uid_to_diff if uid not in all_ids]

        after_diff_dist = dict(zip(*np.unique(self.dataset['difficulty'], return_counts=True)))
        print("\n[DEBUG] Update summary:")
        print(f"  - Total updates attempted: {len(updates)}")
        print(f"  - Successfully updated: {len(updates) - len(missing_ids)}")
        print(f"  - Not found samples: {len(missing_ids)}")
        if missing_ids:
            print(f"    - Missing IDs: {', '.join(missing_ids[:5])}" + 
                ("..." if len(missing_ids) > 5 else ""))

        if len(updates) > 0 and not missing_ids:
            print("\n[DEBUG] Random sample verification:")
            import random
            samples_to_check = min(5, len(updates))
            test_cases = random.sample(updates, samples_to_check)
            for uid, expected_diff in test_cases:
                idx = self.dataset['id'].index(uid)
                actual_diff = self.dataset[idx]['difficulty']
                status = "✓" if actual_diff == expected_diff else "✗"
                print(f"  - ID {uid}: expected {expected_diff}, got {actual_diff} [{status}]")

        return len(missing_ids) == 0

    def _finalize_example(self, example, model_inputs, input_ids, attention_mask, prompt):
        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw"),
                video_grid_thw=model_inputs.get("video_grid_thw"),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                attention_mask=attention_mask,
            )
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length:]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[:self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        example["input_ids"] = input_ids
        example["attention_mask"] = attention_mask
        example["position_ids"] = position_ids
        example["raw_prompt_ids"] = raw_prompt_ids
        example["ground_truth"] = example.pop(self.answer_key)

        return example

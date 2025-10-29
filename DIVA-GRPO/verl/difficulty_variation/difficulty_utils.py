import numpy as np
from PIL import Image, ImageOps
import io
import math
import os
import cv2
from io import BytesIO
import random
from skimage.draw import random_shapes
import numpy as np


def get_edge_color(image, side='left'):
    width, height = image.size
    if side == 'left':
        edge_pixels = [image.getpixel((0, y)) for y in range(height)]
    elif side == 'right':
        edge_pixels = [image.getpixel((width-1, y)) for y in range(height)]
    elif side == 'top':
        edge_pixels = [image.getpixel((x, 0)) for x in range(width)]
    elif side == 'bottom':
        edge_pixels = [image.getpixel((x, height-1)) for x in range(width)]
    else:
        edge_pixels = []
    if edge_pixels:
        avg_color = tuple(
            int(sum(channel)/len(edge_pixels)) 
            for channel in zip(*edge_pixels)
        )
        return avg_color
    return (255, 255, 255)

def rotate_image(input_data, angle, expand=True, fill_color=None):
    try:
        image = Image.open(io.BytesIO(input_data))
    except Exception as e:
        raise ValueError("Unable to parse image data: " + str(e))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    if expand:
        w, h = image.size
        radians = math.radians(angle)
        new_w = int(abs(w * math.cos(radians)) + abs(h * math.sin(radians)))
        new_h = int(abs(w * math.sin(radians)) + abs(h * math.cos(radians)))
    else:
        new_w, new_h = image.size
    if fill_color is None:
        normalized_angle = angle % 360
        if 45 <= normalized_angle < 135:
            fill_color = get_edge_color(image, 'top')
        elif 135 <= normalized_angle < 225:
            fill_color = get_edge_color(image, 'right')
        elif 225 <= normalized_angle < 315:
            fill_color = get_edge_color(image, 'bottom')
        else:
            fill_color = get_edge_color(image, 'left')
    rotated = image.rotate(angle, expand=expand, fillcolor=fill_color)
    if expand:
        rotated = ImageOps.pad(rotated, (new_w, new_h), color=fill_color)
    output = io.BytesIO()
    rotated.save(output, format='PNG')
    return output.getvalue()

def add_gaussian_noise(image_data, noise_intensity=0.1):
    image = Image.open(io.BytesIO(image_data))
    if image is None:
        raise ValueError("Failed to decode image data")
    image = np.array(image)
    if image.dtype != np.float32:
        image = image.astype(np.float32) / 255.0
    noise = np.random.normal(0, noise_intensity, image.shape)
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0, 1)
    noisy_image = (noisy_image * 255).astype(np.uint8)
    _, encoded_image = cv2.imencode('.png', noisy_image)
    return encoded_image.tobytes()

def add_salt_pepper_noise(image_data, noise_strength=0.1):
    if isinstance(image_data, bytes):
        image = np.array(Image.open(io.BytesIO(image_data)))
    else:
        image = image_data
    if image.max() > 1:
        image = image.astype(np.float32) / 255.0
        image = (image > 0.5).astype(np.uint8)
    noisy_image = image.copy()
    h, w = image.shape[:2]
    total_pixels = h * w
    n_noise = int(noise_strength * total_pixels)
    if n_noise == 0:
        return noisy_image
    indices = np.random.choice(total_pixels, size=n_noise, replace=False)
    salt_pixels = n_noise // 2
    pepper_pixels = salt_pixels
    if n_noise % 2 != 0:
        if np.random.rand() > 0.5:
            salt_pixels += 1
        else:
            pepper_pixels += 1
    salt_indices = indices[:salt_pixels]
    noisy_image.flat[salt_indices] = 1
    pepper_indices = indices[salt_pixels:salt_pixels+pepper_pixels]
    noisy_image.flat[pepper_indices] = 0
    noisy_image = (noisy_image * 255).astype(np.uint8)
    _, encoded_image = cv2.imencode('.png', noisy_image)
    return encoded_image.tobytes()

def add_speckle_noise(image_data, noise_intensity=0.1):
    image = Image.open(io.BytesIO(image_data))
    if image is None:
        raise ValueError("Failed to decode image data")
    image = np.array(image)
    if image.dtype != np.float32:
        image = image.astype(np.float32) / 255.0
    noise = np.random.normal(0, noise_intensity, image.shape)
    noisy_image = image + image * noise
    noisy_image = np.clip(noisy_image, 0, 1)
    noisy_image = (noisy_image * 255).astype(np.uint8)
    _, encoded_image = cv2.imencode('.png', noisy_image)
    return encoded_image.tobytes()

def add_random_occlusion(image_data, percent=20, blocks=5):
    image = Image.open(io.BytesIO(image_data))
    if image is None:
        raise ValueError("Failed to decode image data")
    image = np.array(image)
    h, w = image.shape[:2]
    shapes, _ = random_shapes(
        image_shape=(h, w),
        min_shapes=blocks,
        max_shapes=blocks,
        min_size=int(min(h,w)*0.1),
        max_size=int(min(h,w)*0.2),
        random_seed=random.randint(0,1000)
    )
    mask = shapes == 1
    if len(image.shape) == 3:
        mask = np.repeat(mask[:,:,np.newaxis], 3, axis=2)
    occlusion_colors = np.random.randint(
        0, 256, 
        size=(blocks, 1, 1, 3 if len(image.shape)==3 else 1),
        dtype=np.uint8
    )
    for i in range(1, blocks+1):
        shape_mask = (shapes == i)
        if shape_mask.ndim > 2:
            shape_mask = shape_mask.squeeze()
        color = occlusion_colors[i-1]
        for c in range(3):
            image[shape_mask, c] = color[c]
    _, encoded_image = cv2.imencode('.png', image)
    return encoded_image.tobytes()

def add_blur(image_data, blur_intensity=0.1):
    image = Image.open(io.BytesIO(image_data))
    if image is None:
        raise ValueError("Failed to decode image data")
    image = np.array(image)
    if image.dtype != np.uint8:
        if image.dtype == bool:
            image = image.astype(np.uint8) * 255
        else:
            image = image.astype(np.uint8)
    kernel_size = int(blur_intensity * 50)
    kernel_size = max(3, kernel_size)
    kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    _, encoded_image = cv2.imencode('.png', blurred_image)
    return encoded_image.tobytes()

def add_low_resolution(image_data, reduction_factor=0.1):
    image = Image.open(io.BytesIO(image_data))
    if image is None:
        raise ValueError("Failed to decode image data")
    original_size = image.size
    original_mode = image.mode
    new_width = max(1, int(original_size[0] * reduction_factor))
    new_height = max(1, int(original_size[1] * reduction_factor))
    small_image = image.resize((new_width, new_height), Image.NEAREST)
    low_res_image = small_image.resize(original_size, Image.BILINEAR)
    low_res_array = np.array(low_res_image, dtype=np.uint8)
    if original_mode == 'RGB':
        low_res_array = cv2.cvtColor(low_res_array, cv2.COLOR_RGB2BGR)
    if low_res_array.dtype != np.uint8:
        low_res_array = low_res_array.astype(np.uint8)
    _, encoded_image = cv2.imencode('.png', low_res_array)
    return encoded_image.tobytes()

def save_binary_images_to_jpg(image_list, variable_names, output_dir='output'):
    os.makedirs(output_dir, exist_ok=True)
    if len(image_list) != len(variable_names):
        raise ValueError("Image list and variable names length mismatch")
    for i, (image_data, var_name) in enumerate(zip(image_list, variable_names)):
        safe_name = "".join([c if c.isalnum() else "_" for c in var_name])
        filename = f"{safe_name}.jpg"
        filepath = os.path.join(output_dir, filename)
        try:
            with open(filepath, 'wb') as f:
                f.write(image_data)
            print(f"Saved: {filepath}")
        except Exception as e:
            print(f"Failed to save {filepath}: {str(e)}")

def combine_4_images_square_resized(image_binaries, output_size=(1024,1024)):
    if len(image_binaries) != 4:
        raise ValueError("Four images are required")
    images = [Image.open(io.BytesIO(binary)) for binary in image_binaries]
    cell_width = output_size[0] // 2
    cell_height = output_size[1] // 2
    new_image = Image.new('RGB', output_size)
    positions = [(0, 0), (cell_width, 0), (0, cell_height), (cell_width, cell_height)]
    for i, (img, pos) in enumerate(zip(images, positions)):
        img = img.resize((cell_width, cell_height), Image.LANCZOS)
        new_image.paste(img, pos)
    img_byte_arr = io.BytesIO()
    new_image.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()

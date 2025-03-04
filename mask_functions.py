import numpy as np
import torch
import random
import struct
import matplotlib.pyplot as plt
import cv2
from enum import Enum

class DrawMethod(Enum):
    LINE = 'line'
    CIRCLE = 'circle'
    SQUARE = 'square'

def get_random_chunk(data, chunk_size=32):
    w = np.random.randint(0, data.shape[0])
    x = np.random.randint(0, data.shape[1] - chunk_size)
    y = np.random.randint(0, data.shape[2] - chunk_size)
    z = np.random.randint(200, data.shape[3])

    chunk = data[w, x:x + chunk_size, y:y + chunk_size, z]
    norm = np.max(np.abs(chunk))
    if norm != 0:
        chunk /= norm

    return chunk

def random_rotate_and_mirror(chunk):
    k = np.random.randint(0, 4)
    chunk = np.rot90(chunk, k).copy()

    if np.random.randint(0, 2):
        chunk = chunk[::-1, :]

    return chunk

def mask_random_pixels(shape, pixel_dropout=50):

    random_values = np.random.rand(*shape)
    mask = np.less(random_values, pixel_dropout / 100)

    return mask

def mask_random_rows_cols(shape, dropout=50, mode="both"):
    # start as all 0s
    mask = np.zeros(shape, dtype=bool) 

    random_values_rows = np.random.rand(shape[0])
    random_values_cols = np.random.rand(shape[1])

    row_mask = np.less(random_values_rows, dropout / 100)
    col_mask = np.less(random_values_cols, dropout / 100)

    # rows/columns/rows+columns
    if mode in ["rows", "both"]:
        mask[row_mask, :] = True

    if mode in ["cols", "both"]:
        mask[:, col_mask] = True

    return mask

def mask_random_boxes(num_boxes, box_width, box_height, mask_width, mask_height):
    """Create a mask using a given number of randomly-placed rectangles of a given size"""
    max_x = mask_width - box_width
    max_y = mask_height - box_height
    
    mask = np.zeros((mask_height, mask_width), dtype=np.uint8)
    for _ in range(num_boxes):
        x = np.random.randint(0, max_x)
        y = np.random.randint(0, max_y)
        mask[y:y+box_height, x:x+box_width] = 1

    return mask

def make_random_irregular_mask(shape, max_angle=4, max_len=60, max_width=20, min_times=0, max_times=10,
                               draw_method=DrawMethod.LINE):
    """Creates an irregular mask with lines, circles, or squares."""
    height, width = shape
    mask = np.zeros((height, width), np.float32)
    times = np.random.randint(min_times, max_times + 1)
    
    for i in range(times):
        start_x, start_y = np.random.randint(width), np.random.randint(height)
        for j in range(1 + np.random.randint(5)):
            angle = 0.01 + np.random.randint(max_angle)
            angle = 2 * np.pi - angle if i % 2 == 0 else angle
            length = 10 + np.random.randint(max_len)
            brush_w = 5 + np.random.randint(max_width)
            end_x = np.clip(int(start_x + length * np.sin(angle)), 0, width)
            end_y = np.clip(int(start_y + length * np.cos(angle)), 0, height)

            if draw_method == DrawMethod.LINE:
                cv2.line(mask, (start_x, start_y), (end_x, end_y), 1.0, brush_w)
            elif draw_method == DrawMethod.CIRCLE:
                cv2.circle(mask, (start_x, start_y), radius=brush_w, color=1., thickness=-1)
            elif draw_method == DrawMethod.SQUARE:
                radius = brush_w // 2
                mask[max(0, start_y - radius):start_y + radius, max(0, start_x - radius):start_x + radius] = 1

            start_x, start_y = end_x, end_y
    return mask[None, ...]

def make_blurry_mask(shape, blur_size=3):
    """Creates a blurry mask by applying Gaussian blur to a random mask."""
    mask = make_random_irregular_mask(shape)
    blurred_mask = cv2.GaussianBlur(mask[0], (blur_size, blur_size), 1)
    return np.expand_dims(blurred_mask, axis=0)

with open("SegActi-45x201x201x614.bin", "rb") as f:
    # Read dimensions (3 integers as big-endian)
    w = 45
    x = 201
    y = 201
    z = 614
    num_elements = w * x * y * z
    data = np.frombuffer(f.read(num_elements * 4), dtype="f4").reshape(w, x, y, z)

data = np.array(data)
data /= np.max(np.abs(data), axis=(1,2), keepdims=True) + 1e-9

def get_chunk(chunk_size=32):
    return random_rotate_and_mirror(get_random_chunk(data, chunk_size))

def test_seismic_mask():
    chunk = get_chunk(chunk_size=64)  # 64x64
    blurry_mask = make_blurry_mask(chunk.shape)  # Get soft blurry mask

    # Apply the blurry mask as an opacity modifier rather than a binary mask
    masked_chunk = chunk * (1 - blurry_mask[0])  # Scale seismic values by (1 - mask)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(chunk, cmap="seismic")
    axes[0].set_title("Original Seismic Data")
    axes[0].axis("off")

    axes[1].imshow(blurry_mask[0], cmap="gray")  # Show the blurry mask
    axes[1].set_title("Generated Blurry Mask")
    axes[1].axis("off")

    axes[2].imshow(masked_chunk, cmap="seismic", vmin=-1, vmax=1)  # Show masked data with soft effect
    axes[2].set_title("Soft Masked Seismic Data")
    axes[2].axis("off")

    plt.show()

if __name__ == "__main__":
    test_seismic_mask()
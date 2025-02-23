import numpy as np
import torch
import random
import struct
import matplotlib.pyplot as plt

def get_random_chunk(data, chunk_size=32):
    w = np.random.randint(0, data.shape[0])
    x = np.random.randint(0, data.shape[1] - chunk_size)
    y = np.random.randint(0, data.shape[2] - chunk_size)
    z = np.random.randint(200, data.shape[3])

    return data[w, x:x + chunk_size, y:y + chunk_size, z]

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

# Example usage:
mask_width = 100
mask_height = 100
box_width = 20
box_height = 20
num_boxes = 5

mask = mask_random_boxes(num_boxes, box_width, box_height, mask_width, mask_height)
plt.imshow(mask, cmap='gray')
plt.title('Generated Mask')
plt.show()

'''
# Test parameters
shape = (32, 32)  
pixel_dropout = 25

# Generate the mask
mask = mask_random_pixels(shape, pixel_dropout)

# Plot the mask
plt.figure(figsize=(5, 5))
plt.imshow(mask, cmap="gray")
plt.title(f"{pixel_dropout}%")
plt.colorbar()
plt.show()
'''
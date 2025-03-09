import cv2
import numpy as np
import random
from enum import Enum

class DrawMethod(Enum):
    LINE = 'line'
    CIRCLE = 'circle'
    SQUARE = 'square'


#customizable mask to cover approx 1/3-1/2(sometimes a bit lower) 
def make_seismic_mask(shape, max_len=None, max_width=None, min_times=5, max_times=8,
                       draw_method=DrawMethod.LINE):
    height, width = shape
    mask = np.zeros((height, width), np.float32)  # Changed to float32 for OpenCV compatibility
    times = np.random.randint(min_times, max_times + 1)

    if max_len == None:
        max_len = min(width, height) // 3

    if max_width == None:
        max_width = min(width, height) // 4

    min_len = min(width, height) // 10
    min_width = min(width, height) // 10

    for i in range(times):
        start_x, start_y = np.random.randint(width), np.random.randint(height)
        for j in range(1 + np.random.randint(3)):
            angle = 0.01 + np.random.randint(4)
            angle = 2 * np.pi - angle if i % 2 == 0 else angle
            length = min_len + np.random.randint(max_len)
            brush_w = min_width + np.random.randint(max_width - min_width)
            end_x = np.clip(int(start_x + length * np.sin(angle)), 0, width-1)
            end_y = np.clip(int(start_y + length * np.cos(angle)), 0, height-1)

            if draw_method == DrawMethod.LINE:
                cv2.line(mask, (start_x, start_y), (end_x, end_y), 1, brush_w)
            elif draw_method == DrawMethod.CIRCLE:
                cv2.circle(mask, (start_x, start_y), radius=brush_w, color=1, thickness=-1)
            elif draw_method == DrawMethod.SQUARE:
                radius = brush_w // 2
                mask[max(0, start_y - radius):min(height, start_y + radius), 
                     max(0, start_x - radius):min(width, start_x + radius)] = 1

            start_x, start_y = end_x, end_y

    return (mask > 0).astype(bool)

def make_seismic_masks(num_masks, *args, **kvargs):
    masks = np.stack([make_seismic_mask(*args, **kvargs) for _ in range(num_masks)])
    return np.expand_dims(masks, 1)

#creates blurry mask
def make_blurry_mask(shape, blur_size=3, **kwargs):
    mask = make_seismic_mask(shape, **kwargs)
    mask_float = mask.astype(np.float32)#convert to float to blur
    blurred_mask = cv2.GaussianBlur(mask_float, (blur_size, blur_size), 1)
    return blurred_mask


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Example size for mask
    # shape = (201, 201)
    shape = (64, 64)
    
    
    mask = make_seismic_mask(shape)
    
    # Calculate coverage
    coverage = np.mean(mask)
    

    plt.figure(figsize=(6, 6))
    plt.imshow(mask, cmap='gray')
    plt.title(f'Seismic Mask (Coverage: {coverage*100:.1f}%)')
    plt.axis('off')
    plt.show()

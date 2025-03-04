import cv2
import numpy as np
import random
from enum import Enum

class DrawMethod(Enum):
    LINE = 'line'
    CIRCLE = 'circle'
    SQUARE = 'square'

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

# Example usage
if __name__ == "__main__":
    shape = (64, 64)  # Example mask size
    mask = make_random_irregular_mask(shape)
    blurry_mask = make_blurry_mask(shape)

    import matplotlib.pyplot as plt
    plt.subplot(1, 2, 1)
    plt.imshow(mask[0], cmap='gray')
    plt.title('Irregular Mask')
    
    plt.subplot(1, 2, 2)
    plt.imshow(blurry_mask[0], cmap='gray')
    plt.title('Blurry Mask')
    
    plt.show()
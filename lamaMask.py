import cv2
import numpy as np
import random
from enum import Enum

class DrawMethod(Enum):
    LINE = 'line'
    CIRCLE = 'circle'
    SQUARE = 'square'

def make_seismic_mask(shape, max_len=60, max_width=30, min_times=5, max_times=8,
                       draw_method=DrawMethod.LINE):
    """Creates an irregular mask for seismic data infilling.
    Masks approximately 1/3 to 1/2 of the image with wide lines.
    """
    height, width = shape
    mask = np.zeros((height, width), np.float32)  # Changed to float32 for OpenCV compatibility
    times = np.random.randint(min_times, max_times + 1)
    
    for i in range(times):
        start_x, start_y = np.random.randint(width), np.random.randint(height)
        for j in range(1 + np.random.randint(3)):
            angle = 0.01 + np.random.randint(4)
            angle = 2 * np.pi - angle if i % 2 == 0 else angle
            length = 20 + np.random.randint(max_len)
            brush_w = 15 + np.random.randint(max_width - 15)
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
    
    # Convert to boolean mask after drawing
    return (mask > 0).astype(bool)

# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Example mask size 
    shape = (201, 201)
    
    # Create a mask
    mask = make_seismic_mask(shape)
    
    # Calculate coverage
    coverage = np.mean(mask)
    
    # Plot mask
    plt.figure(figsize=(6, 6))
    plt.imshow(mask, cmap='gray')
    plt.title(f'Seismic Mask (Coverage: {coverage*100:.1f}%)')
    plt.axis('off')
    plt.show()
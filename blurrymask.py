import numpy as np
import cv2
import random
from enum import Enum

class DrawMethod(Enum):
    LINE = 'line'
    CIRCLE = 'circle'
    SQUARE = 'square'

def make_random_irregular_mask(shape, max_angle=4, max_len=60, max_width=20, min_times=0, max_times=10,
                               draw_method=DrawMethod.LINE):
    draw_method = DrawMethod(draw_method)
    height, width = shape
    mask = np.zeros((height, width), np.float32)
    times = np.random.randint(min_times, max_times + 1)
    for i in range(times):
        start_x = np.random.randint(width)
        start_y = np.random.randint(height)
        for j in range(1 + np.random.randint(5)):
            angle = 0.01 + np.random.randint(max_angle)
            length = 10 + np.random.randint(max_len)
            brush_w = 5 + np.random.randint(max_width)
            end_x = np.clip(int(start_x + length * np.sin(angle)), 0, width - 1)
            end_y = np.clip(int(start_y + length * np.cos(angle)), 0, height - 1)
            if draw_method == DrawMethod.LINE:
                cv2.line(mask, (start_x, start_y), (end_x, end_y), 1.0, brush_w)
            elif draw_method == DrawMethod.CIRCLE:
                cv2.circle(mask, (start_x, start_y), radius=brush_w, color=1.0, thickness=-1)
            elif draw_method == DrawMethod.SQUARE:
                radius = brush_w // 2
                mask[max(0, start_y - radius):min(height, start_y + radius),
                     max(0, start_x - radius):min(width, start_x + radius)] = 1
            start_x, start_y = end_x, end_y
    return mask[None, ...]

def make_random_rectangle_mask(shape, margin=10, bbox_min_size=30, bbox_max_size=100, min_times=0, max_times=3):
    height, width = shape
    mask = np.zeros((height, width), np.float32)
    bbox_max_size = min(bbox_max_size, height - margin * 2, width - margin * 2)
    times = np.random.randint(min_times, max_times + 1)
    for _ in range(times):
        box_width = np.random.randint(bbox_min_size, bbox_max_size)
        box_height = np.random.randint(bbox_min_size, bbox_max_size)
        start_x = np.random.randint(margin, width - margin - box_width + 1)
        start_y = np.random.randint(margin, height - margin - box_height + 1)
        mask[start_y:start_y + box_height, start_x:start_x + box_width] = 1
    return mask[None, ...]

def make_random_superres_mask(shape, min_step=2, max_step=4, min_width=1, max_width=3):
    height, width = shape
    mask = np.zeros((height, width), np.float32)
    step_x = np.random.randint(min_step, max_step + 1)
    width_x = np.random.randint(min_width, min(step_x, max_width + 1))
    offset_x = np.random.randint(0, step_x)
    step_y = np.random.randint(min_step, max_step + 1)
    width_y = np.random.randint(min_width, min(step_y, max_width + 1))
    offset_y = np.random.randint(0, step_y)
    for dy in range(width_y):
        mask[offset_y + dy::step_y] = 1
    for dx in range(width_x):
        mask[:, offset_x + dx::step_x] = 1
    return mask[None, ...]

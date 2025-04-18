import torch
import torch.nn as nn

import numpy as np
import draw_mask
from model import CombinedGenerator, device, resolution

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# G = CombinedGenerator("trained_128x128_b54a594.ckpt").to(device)
G = CombinedGenerator("trained_64x64_cbbfce7_div1.ckpt").to(device)

def center_of_mass_and_rectangle(mask, rect_size):
    """
    Given a 2D binary NumPy array 'mask' and a rectangle size (width, height),
    this function computes the center of mass of the 'mask' and returns the
    edge coordinates (top, left, bottom, right) of a rectangle of the given size 
    centered at that location.

    Parameters:
    - mask (np.ndarray): 2D binary numpy array.
    - rect_size (tuple): (width, height) of the desired rectangle.

    Returns:
    - tuple: (top, left, bottom, right) edge coordinates of the rectangle.
             The coordinates are integers.
    """
    if mask.ndim != 2:
        raise ValueError("Input mask must be a 2D array.")
        
    height, width = mask.shape
    rect_width, rect_height = rect_size
    
    # Get indices where mask is non-zero (assuming binary mask with 1s marking the region of interest)
    indices = np.argwhere(mask)
    if indices.size == 0:
        raise ValueError("The mask does not contain any nonzero elements.")
        
    # Calculate the center of mass.
    # axis 0 corresponds to row (y-coordinate) and axis 1 corresponds to column (x-coordinate)
    center_y, center_x = np.mean(indices, axis=0)
    
    # Convert center coordinates to integers (rounding to nearest integer)
    center_x = int(round(center_x))
    center_y = int(round(center_y))
    
    # Compute half sizes.
    half_width = rect_width // 2
    half_height = rect_height // 2
    
    # Calculate the rectangle edges.
    # For even dimensions, the rectangle will be slightly off center if strict symmetry is required.
    left = max(center_x - half_width, 0)
    right = min(center_x + half_width + (rect_width % 2), width)  # Adjust for odd widths.
    top = max(center_y - half_height, 0)
    bottom = min(center_y + half_height + (rect_height % 2), height)  # Adjust for odd heights.
    
    # If the rectangle exceeds the boundary, adjust to fit within the mask:
    if right - left < rect_width:
        # Adjust horizontally if needed.
        if left == 0:
            right = min(rect_width, width)
        elif right == width:
            left = max(width - rect_width, 0)
    
    if bottom - top < rect_height:
        # Adjust vertically if needed.
        if top == 0:
            bottom = min(rect_height, height)
        elif bottom == height:
            top = max(height - rect_height, 0)
    
    return (top, left, bottom, right)

def match_shape(tensor, target):
    # Pad or crop tensor to match the target shape
    _, _, h, w = tensor.shape
    _, _, H, W = target.shape
    dh, dw = H - h, W - w
    if dh > 0 or dw > 0:
        tensor = nn.functional.pad(tensor, [0, dw, 0, dh])
    elif dh < 0 or dw < 0:
        tensor = tensor[:, :, :H, :W]
    return tensor

def infill_and_display(model, masked_volume, mask):
    model.eval()
    infilled_volume = masked_volume.copy()
    top, left, bottom, right = center_of_mass_and_rectangle(mask, resolution)

    with torch.no_grad():
        for i in range(masked_volume.shape[2]):
            slice_input = masked_volume[:, :, i]
            #slice_mask = mask[:, :, i]

            if not np.any(mask):
                continue  # Skip if no masked region

            cropped_input = slice_input[top:bottom, left:right]
            cropped_mask = mask[top:bottom, left:right]

            # Convert to tensors
            input_tensor = torch.tensor(cropped_input, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            mask_tensor = torch.tensor(cropped_mask, dtype=torch.bool, device=device).unsqueeze(0).unsqueeze(0)

            # Run model
            output = model(input_tensor, mask_tensor).squeeze().cpu().numpy()

            # Infill only masked region
            infilled_volume[top:bottom, left:right, i][cropped_mask] = output[cropped_mask]

            print(f"Infilled slice {i + 1} of {masked_volume.shape[2]}")

    draw_mask.show_volume_with_slider(masked_volume, mask, infilled_volume)

if __name__ == "__main__":
    model = G  # Generator already loaded
        
    with open("SegActi-45x201x201x614.bin", "rb") as f:
        w, x, y, z = 45, 201, 201, 614
        data = np.frombuffer(f.read(w * x * y * z * 4), dtype="f4").reshape(w, x, y, z)

    volume = draw_mask.choose_volume(data)
    masked_volume, mask = draw_mask.apply_mask(volume)
    #print("mask applied.")
    infill_and_display(model, masked_volume, mask)

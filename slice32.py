import numpy as np
import torch
import random
import struct
import matplotlib.pyplot as plt

def get_random_masked_slice(data, mask_size=16):
    """
    Extracts a random 32x32 slice from a 3D volume in a random orientation, applies a mask,
    and randomly rotates the slice.

    Parameters:
    - data (np.ndarray): The 3D volumetric data (z, y, x).
    - mask_size (int): Size of the square mask applied in the center of the slice.

    Returns:
    - masked_slice (torch.Tensor): The 32x32 slice with applied mask.
    - mask (torch.Tensor): The binary mask (1 for valid pixels, 0 for masked area).
    - orientation (str): The selected orientation plane.
    - rotation_angle (int): The applied rotation in degrees.
    """
    assert data.ndim == 3, "Input data must be a 3D numpy array."

    orientation = random.choice(["XY", "ZX", "ZY"])

    if orientation == "XY":
        z_idx = np.random.randint(0, data.shape[0])
        slice_2d = data[z_idx, :, :]
    elif orientation == "ZX":
        y_idx = np.random.randint(0, data.shape[1])
        slice_2d = data[:, y_idx, :]
    else:  
        x_idx = np.random.randint(0, data.shape[2])
        slice_2d = data[:, :, x_idx]

    H, W = slice_2d.shape
    target_size = 32

    if H < target_size or W < target_size:
        pad_h = max(0, target_size - H)
        pad_w = max(0, target_size - W)
        slice_2d = np.pad(slice_2d, ((0, pad_h), (0, pad_w)), mode='constant')
    elif H > target_size or W > target_size:
        h_start = (H - target_size) // 2
        w_start = (W - target_size) // 2
        slice_2d = slice_2d[h_start:h_start + target_size, w_start:w_start + target_size]

    # Randomly rotate the slice by 0, 90, 180, or 270 degrees
    rotation_angle = random.choice([0, 90, 180, 270])
    k = rotation_angle // 90  
    slice_2d = np.rot90(slice_2d, k).copy()  # Copy ensures compatibility with PyTorch

    slice_tensor = torch.tensor(slice_2d, dtype=torch.float32).unsqueeze(0)

    mask = torch.ones_like(slice_tensor)
    center = target_size // 2
    half_size = mask_size // 2
    mask[:, center - half_size:center + half_size, center - half_size:center + half_size] = 0

    masked_slice = slice_tensor * mask

    return masked_slice, mask, orientation, rotation_angle

# Read binary file
with open("TestData.bin", "rb") as f:
    shape = struct.unpack("iii", f.read(12))  
    z, x, y = shape  
    num_elements = z * y * x  
    data = np.frombuffer(f.read(num_elements * 4), dtype="f4").reshape(y, x, z)
    rms = struct.unpack("f", f.read(4))[0]  

print("Loaded 3D array shape:", data.shape)

# rand mask slice
masked_slice, mask, orientation, rotation_angle = get_random_masked_slice(data)

print(f"Masked Slice Shape: {masked_slice.shape}, Orientation: {orientation}, Rotation: {rotation_angle}°")

# imaging
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(masked_slice.squeeze(), cmap='gray')
plt.title(f"Masked Slice ({orientation} plane, {rotation_angle}°)")

plt.subplot(1, 2, 2)
plt.imshow(mask.squeeze(), cmap='gray')
plt.title("Mask")

plt.show()

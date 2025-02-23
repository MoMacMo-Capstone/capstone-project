import numpy as np
import struct

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

with open("SegActi-45x201x201x614.bin", "rb") as f:
    w = 45
    x = 201
    y = 201
    z = 614
    num_elements = w * x * y * z
    data = np.frombuffer(f.read(num_elements * 4), dtype="f4").reshape(w, x, y, z)
    data = np.array(data)
    data /= np.max(np.abs(data), axis=(1,2), keepdims=True) + 1e-9


def get_random_chunk(data, chunk_size=32, min_z=200):
    """
    Select a random 32x32 spatial chunk from a 4D array, only from deeper sections.
    
    Parameters:
    -----------
    data : np.ndarray
        4D array with shape (w, x, y, z)
    chunk_size : int
        Size of the spatial chunk to extract (default: 32)
    min_z : int
        Minimum z-index to consider (default: 200)
        
    Returns:
    --------
    chunk : np.ndarray
        4D array with shape (w, chunk_size, chunk_size, z)
    x_start : int
        Starting x index of the chunk
    y_start : int
        Starting y index of the chunk
    z_start : int
        Starting z index of the chunk
    """
    # Get array dimensions
    w, x, y, z = data.shape
    
    # Ensure min_z is valid
    if min_z >= z:
        raise ValueError(f"min_z ({min_z}) must be less than z dimension size ({z})")
    
    # Calculate valid ranges for random selection
    max_x = x - chunk_size + 1
    max_y = y - chunk_size + 1
    max_z = z  # We'll use the full remaining depth
    
    # Generate random starting points
    x_start = np.random.randint(0, max_x)
    y_start = np.random.randint(0, max_y)
    z_start = np.random.randint(min_z, max_z)
    
    # Extract the chunk
    chunk = data[:, x_start:x_start+chunk_size, y_start:y_start+chunk_size, z_start:]
    
    return chunk, x_start, y_start, z_start

def rotate_chunk(chunk):
    """
    Randomly rotate a chunk in one of 8 possible ways, then possibly mirror along any axis.
    
    Parameters:
    -----------
    chunk : np.ndarray
        4D array with shape (w, chunk_size, chunk_size, z)
        
    Returns:
    --------
    rotated_chunk : np.ndarray
        Rotated and possibly mirrored version of the input chunk
    transformation : str 
        Description of the applied transformations
    """
    # Choose random rotation (0, 90, 180, or 270 degrees)
    k_rotations = np.random.randint(0, 4)  # Number of 90-degree rotations
    
    # Choose random flip (True or False)
    do_flip = np.random.choice([True, False])
    
    # Create a copy to avoid modifying the original
    transformed = chunk.copy()
    
    # Apply rotation (rotate around the spatial dimensions)
    if k_rotations > 0:
        # For each w and z slice, rotate the x-y plane
        for w_idx in range(chunk.shape[0]):
            for z_idx in range(chunk.shape[3]):
                transformed[w_idx, :, :, z_idx] = np.rot90(
                    chunk[w_idx, :, :, z_idx], 
                    k=k_rotations
                )
    
    # Apply initial flip if selected
    if do_flip:
        transformed = np.flip(transformed, axis=2)  # Flip along y-axis
    
    # Create description of the initial transformation
    rotation_degrees = k_rotations * 90
    flip_text = " and flipped" if do_flip else ""
    transformation = f"Rotated {rotation_degrees}°{flip_text}"
    
    # 50% chance for additional mirroring
    if np.random.choice([True, False]):
        # Choose which axis to mirror along (0=w, 1=x, 2=y, 3=z)
        mirror_axis = np.random.randint(0, 4)
        transformed = np.flip(transformed, axis=mirror_axis)
        
        # Add mirroring description
        axis_names = {0: 'w', 1: 'x', 2: 'y', 3: 'z'}
        transformation += f", mirrored along {axis_names[mirror_axis]}-axis"
    
    return transformed, transformation

# Example usage:
# Get a deep chunk
chunk, x_start, y_start, z_start = get_random_chunk(data)
print(f"Original chunk shape: {chunk.shape}")
print(f"Chunk location: x={x_start}:{x_start+32}, y={y_start}:{y_start+32}, z={z_start}:end")

# Rotate the chunk
rotated_chunk, transformation = rotate_chunk(chunk)
print(f"Transformation applied: {transformation}")
print(f"Rotated chunk shape: {rotated_chunk.shape}")

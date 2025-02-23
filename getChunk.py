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


def get_random_chunk(data, chunk_size=32):
    """
    Select a random 32x32 spatial chunk from a 4D array.
    
    Parameters:
    -----------
    data : np.ndarray
        4D array with shape (w, x, y, z)
    chunk_size : int
        Size of the spatial chunk to extract (default: 32)
        
    Returns:
    --------
    chunk : np.ndarray
        4D array with shape (w, chunk_size, chunk_size, z)
    x_start : int
        Starting x index of the chunk
    y_start : int
        Starting y index of the chunk
    """
    # Get array dimensions
    w, x, y, z = data.shape
    
    # Calculate valid ranges for random selection
    # Ensure we don't go out of bounds
    max_x = x - chunk_size + 1
    max_y = y - chunk_size + 1
    
    # Generate random starting points
    x_start = np.random.randint(0, max_x)
    y_start = np.random.randint(0, max_y)
    
    # Extract the chunk
    chunk = data[:, x_start:x_start+chunk_size, y_start:y_start+chunk_size, :]
    
    return chunk, x_start, y_start

# Set random seed for reproducibility (optional)
np.random.seed(420)

# Get a random chunk
chunk, x_start, y_start = get_random_chunk(data)

print(f"Chunk shape: {chunk.shape}")
print(f"Chunk location: x={x_start}:{x_start+32}, y={y_start}:{y_start+32}")
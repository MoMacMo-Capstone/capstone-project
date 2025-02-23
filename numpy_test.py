'''
Created on Feb 14, 2025

@author: chuck and nathan
'''
import numpy as np
import struct

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Read binary file
with open("SegActi-45x201x201x614.bin", "rb") as f:
    # Read dimensions (3 integers as big-endian)
    w = 45
    x = 201
    y = 201
    z = 614
    num_elements = w * x * y * z
    data = np.frombuffer(f.read(num_elements * 4), dtype="f4").reshape(w, x, y, z)

print("Loaded 3D array shape:", data.shape)
print("Computed RMS:", np.sqrt(np.mean(data**2)))  # Compute RMS from loaded data

# Load 3D float array from binary file (assuming you've already read it)
# `data` is a (z, y, x) NumPy array

# Set up figure and subplots
# fig, axs = plt.subplots(1, 3, figsize=(12, 4))
# plt.subplots_adjust(bottom=0.25)  # Leave space for sliders

data = np.array(data)
data /= np.max(np.abs(data), axis=(1,2), keepdims=True) + 1e-9

plt.imshow(data[42, :, :, 450], cmap="gray", aspect="auto")
plt.show()

# Initial slice indices (middle slices)
# z_idx = data.shape[0] // 2
# y_idx = data.shape[1] // 2
# x_idx = data.shape[2] // 2

# # Display slices
# zx_plot = axs[0].imshow(data[:, y_idx, :], cmap="gray", aspect="auto")
# zy_plot = axs[1].imshow(data[:, :, x_idx], cmap="gray", aspect="auto")
# xy_plot = axs[2].imshow(data[z_idx, :, :], cmap="gray", aspect="auto")

# # Titles
# axs[0].set_title("ZX Plane (y fixed)")
# axs[1].set_title("ZY Plane (x fixed)")
# axs[2].set_title("XY Plane (z fixed)")

# # Add sliders for interactive navigation
# ax_slider_z = plt.axes([0.2, 0.1, 0.6, 0.03])
# ax_slider_y = plt.axes([0.2, 0.06, 0.6, 0.03])
# ax_slider_x = plt.axes([0.2, 0.02, 0.6, 0.03])

# slider_z = Slider(ax_slider_z, "Z", 0, data.shape[0] - 1, valinit=z_idx, valstep=1)
# slider_y = Slider(ax_slider_y, "Y", 0, data.shape[1] - 1, valinit=y_idx, valstep=1)
# slider_x = Slider(ax_slider_x, "X", 0, data.shape[2] - 1, valinit=x_idx, valstep=1)

# # Update function for sliders
# def update(val):
    # z = int(slider_z.val)
    # y = int(slider_y.val)
    # x = int(slider_x.val)

    # zx_plot.set_data(data[:, y, :])  # Update ZX slice
    # zy_plot.set_data(data[:, :, x])  # Update ZY slice
    # xy_plot.set_data(data[z, :, :])  # Update XY slice

    # fig.canvas.draw_idle()

# # Connect sliders to update function
# slider_z.on_changed(update)
# slider_y.on_changed(update)
# slider_x.on_changed(update)

# plt.show()

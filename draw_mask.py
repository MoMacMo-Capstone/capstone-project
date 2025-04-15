import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchmetrics.image.fid import FrechetInceptionDistance

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector, Slider    # drawing
from matplotlib.path import Path                # drawing

def choose_volume(data):
    print(f"Available volumes: 0 to {data.shape[0]-1}")
    while True:
        try:
            volume_idx = int(input("Enter the volume index to use: "))
            if 0 <= volume_idx < data.shape[0]:
                return data[volume_idx]
            else:
                print("Invalid index, try again.")
        except ValueError:
            print("Please enter a valid integer.")

def apply_mask(volume):
    mask = make_drawn_mask_3d(volume)
    masked_volume = volume.copy()
    masked_volume[mask] = 0  # apply mask
    return masked_volume, mask

def show_volume_with_slider(masked_volume, mask_volume, infilled_volume):
    """
    Displays 3 volumes side-by-side using a slider to scroll through slices.
    """
    num_slices = masked_volume.shape[2]
    slice_idx = num_slices // 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plt.subplots_adjust(bottom=0.2)

    titles = ["Image", "Mask", "Infill"]
    imgs = [
        axes[0].imshow(masked_volume[:, :, slice_idx], cmap='gray'),
        axes[1].imshow(mask_volume[:, :, slice_idx], cmap='gray'),
        axes[2].imshow(infilled_volume[:, :, slice_idx], cmap='gray')
    ]

    for ax, title in zip(axes, titles):
        ax.set_title(title)
        ax.axis('off')

    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, 'Slice', 0, num_slices - 1, valinit=slice_idx, valfmt='%0.0f')

    def update(val):
        idx = int(slider.val)
        imgs[0].set_data(masked_volume[:, :, idx])
        imgs[1].set_data(mask_volume[:, :, idx])
        imgs[2].set_data(infilled_volume[:, :, idx])
        for i, ax in enumerate(axes):
            ax.set_title(f"{titles[i]} - Slice {idx}")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()

class MaskDrawer:
    def __init__(self, slice_2d):
        self.slice = slice_2d
        self.mask = np.zeros_like(slice_2d, dtype=bool)
        self.fig, self.ax = plt.subplots()
        self.canvas = self.ax.imshow(slice_2d, cmap='gray')
        self.lasso = LassoSelector(self.ax, onselect=self.on_select)
        self.finished = False

        print("Draw a mask with your mouse. Close the window when done.")
        plt.show()

    def on_select(self, verts):
        path = Path(verts)
        y, x = np.meshgrid(np.arange(self.slice.shape[1]), np.arange(self.slice.shape[0]))
        coords = np.vstack((y.ravel(), x.ravel())).T  # (col, row) order
        self.mask = path.contains_points(coords).reshape(self.slice.shape)
        self.canvas.set_data(np.where(self.mask, 1.0, self.slice))
        self.fig.canvas.draw_idle()

    def get_mask(self):
        return self.mask

def make_drawn_mask_3d(volume):
    """
    Opens a UI for the user to draw a mask on one slice,
    then applies the mask across all slices.
    """
    middle_idx = volume.shape[2] // 2
    slice_2d = volume[:, :, middle_idx]
    drawer = MaskDrawer(slice_2d)
    mask_2d = drawer.get_mask()
    mask_3d = np.repeat(mask_2d[:, :, np.newaxis], volume.shape[2], axis=2)
    return mask_3d


'''
def infill_and_display(model, masked_volume, mask):
    model.eval()
    infilled_volume = masked_volume.copy()

    with torch.no_grad():
        for i in range(masked_volume.shape[2]):
            slice_input = masked_volume[:, :, i]
            slice_mask = mask[:, :, i]

            input_tensor = torch.tensor(slice_input, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            mask_tensor = torch.tensor(slice_mask, dtype=torch.bool).unsqueeze(0).unsqueeze(0)

            # Resize to 64x64
            input_resized = nn.functional.interpolate(input_tensor, size=(64, 64), mode='bilinear', align_corners=False)
            mask_resized = nn.functional.interpolate(mask_tensor.float(), size=(64, 64), mode='nearest').bool()

            stdev_val = torch.std(input_resized[mask_resized == 0]) if (mask_resized == 0).any() else torch.tensor(1.0)
            stdev_tensor = torch.full_like(input_resized, stdev_val)

            # Infill using resized data
            output = model(input_resized, stdev_tensor, mask_resized).squeeze(0).cpu()

            # Resize output back to original resolution
            output_resized = nn.functional.interpolate(output.unsqueeze(0), size=input_tensor.shape[-2:], mode='bilinear', align_corners=False).squeeze().numpy()

            # Infill only masked region
            infilled_volume[:, :, i][slice_mask] = output_resized[slice_mask]
    
    draw_mask.show_volume_with_slider(volume, mask, infilled_volume)
'''
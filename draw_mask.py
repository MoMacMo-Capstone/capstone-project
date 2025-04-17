import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector, Slider    # drawing
from matplotlib.path import Path                # drawing

def rms_norm(image):
    return np.sqrt(np.mean(np.square(image), axis=(0, 1), keepdims=True)) + 1e-9

def rms_normalize(image):
    return image / rms_norm(image)

def color_images(image):
    norm_image = rms_normalize(image) * 1
    stacked = np.stack([
        norm_image,
        np.abs(norm_image) - 1,
        -norm_image
    ], axis=2)
    return np.clip(stacked, 0, 1)

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
    mask = make_drawn_mask_2d(volume)
    masked_volume = volume.copy()
    masked_volume[mask] = 0  # apply mask
    return masked_volume, mask

def show_volume_with_slider(masked_volume, mask, infilled_volume):
    """
    Displays 3 volumes side-by-side using a slider to scroll through slices.
    The infill image is colorized using `color_images`.
    """
    num_slices = masked_volume.shape[2]
    slice_idx = num_slices // 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plt.subplots_adjust(bottom=0.2)

    # Initial colorized infill
    initial_infill_slice = infilled_volume[:, :, slice_idx]
    color_infill = color_images(initial_infill_slice)  # shape [H, W, 3]
    color_image = color_images(masked_volume[:, :, slice_idx])

    titles = ["Image", "Mask", "Infill (Colorized)"]
    imgs = [
        axes[0].imshow(color_image),
        axes[1].imshow(mask, cmap='gray'),
        axes[2].imshow(color_infill)
    ]

    for ax, title in zip(axes, titles):
        ax.set_title(title)
        ax.axis('off')

    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, 'Slice', 0, num_slices - 1, valinit=slice_idx, valfmt='%0.0f')

    def update(val):
        idx = int(slider.val)

        # Update grayscale masked image and mask
        imgs[1].set_data(mask)
        color_image = color_images(masked_volume[:, :, slice_idx])

        # Recolor infill slice
        infill_slice = infilled_volume[:, :, idx]
        masked_slice = masked_volume[:, :, idx]
        color_infill = color_images(infill_slice)
        color_image = color_images(masked_slice)
        imgs[2].set_data(color_infill)
        imgs[0].set_data(color_image)

        for i, ax in enumerate(axes):
            ax.set_title(f"{titles[i]} - Slice {idx}")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()

class MaskDrawer:
    def __init__(self, slice_2d):
        # Colorize the input slice for visualization
        color_slice = color_images(slice_2d)  # [H, W, 3]
        self.color_slice = color_slice
        self.slice = slice_2d
        self.mask = np.zeros_like(slice_2d, dtype=bool)

        self.fig, self.ax = plt.subplots()
        self.canvas = self.ax.imshow(color_slice)
        self.lasso = LassoSelector(self.ax, onselect=self.on_select)
        self.finished = False

        print("Draw a mask with your mouse. Close the window when done.")
        plt.show()

    def on_select(self, verts):
        path = Path(verts)
        y, x = np.meshgrid(np.arange(self.slice.shape[1]), np.arange(self.slice.shape[0]))
        coords = np.vstack((y.ravel(), x.ravel())).T
        self.mask = path.contains_points(coords).reshape(self.slice.shape)

        # Visual feedback: blend color with white in masked areas
        masked_color = self.color_slice.copy()
        masked_color[self.mask] = [1.0, 1.0, 1.0]  # white mask feedback

        self.canvas.set_data(masked_color)
        self.fig.canvas.draw_idle()

    def get_mask(self):
        return self.mask

def make_drawn_mask_2d(volume):
    """
    Opens a UI for the user to draw a mask on one slice,
    then applies the mask across all slices.
    """
    middle_idx = volume.shape[2] // 2
    slice_2d = volume[:, :, middle_idx]
    drawer = MaskDrawer(slice_2d)
    mask_2d = drawer.get_mask()
    #mask_3d = np.repeat(mask_2d[:, :, np.newaxis], volume.shape[2], axis=2)
    return mask_2d

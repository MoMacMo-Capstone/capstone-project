import torch
import torch.nn as nn
import torch.optim as optim
# from torch.utils.data import DataLoader
#from torch.utils.tensorboard import SummaryWriter
import torchvision
# from torchvision import datasets, transforms
from torchmetrics.image.fid import FrechetInceptionDistance

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector    # drawing
from matplotlib.path import Path                # drawing
# import lama_mask
import draw_mask
from model import CombinedGenerator

def pink_noise(shape):
    y = torch.fft.fftfreq(shape[2], device=device).view((1, 1, -1, 1))
    x = torch.fft.fftfreq(shape[3], device=device).view((1, 1, 1, -1))
    radial_frequency = torch.sqrt(x * x + y * y)
    noise = torch.rand(*shape, device=device)
    noise = torch.fft.fft2(noise)
    noise /= radial_frequency
    noise[:, :, 0, 0] = 0
    return torch.fft.ifft2(noise).abs()

def leaky_relu(z):
    return nn.functional.leaky_relu(z, 0.2)

# def noise_to_inpainting(noise, mean, stdev, mask):
    # inpainting = noise * stdev + mean
    # return torch.where(mask, inpainting, mean)

# def inpainting_to_noise(image, mean, stdev, mask):
    # noise = ((image - mean) / stdev).clamp(-3, 3)
    # return noise.masked_fill(~mask, 0)

def abs_norm(images):
    return images.abs().amax(2, keepdim=True).amax(3, keepdim=True) + 1e-9

def abs_normalize(images):
    return images / abs_norm(images)

def color_images(images):
    images = abs_normalize(images) * 2
    return torch.cat([images, images.abs() - 1, -images], dim=1).clamp(0, 1)

class AOTBlock(nn.Module):
    def __init__(self, channels):
        super(AOTBlock, self).__init__()
        self.l1_d1 = nn.Conv2d(channels, channels // 4, 3, dilation=1, padding=1)
        self.l1_d2 = nn.Conv2d(channels, channels // 4, 3, dilation=2, padding=2)
        self.l1_d4 = nn.Conv2d(channels, channels // 4, 3, dilation=4, padding=4)
        self.l1_d8 = nn.Conv2d(channels, channels // 4, 3, dilation=8, padding=8)
        self.l2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, z):
        orig = z
        z = torch.cat([l(z) for l in [self.l1_d1, self.l1_d2, self.l1_d4, self.l1_d8]], dim=1)
        z = leaky_relu(z)
        z = self.l2(z)
        return z + orig

class UNET(nn.Module):
    def __init__(self, channels, resolution, inject_noise=False):
        super(UNET, self).__init__()
        self.block1 = nn.Sequential(
            AOTBlock(channels),
            AOTBlock(channels),
        )
        self.block2 = nn.Sequential(
            AOTBlock(channels),
            AOTBlock(channels),
        )

        if inject_noise:
            self.noise_injector = nn.Conv2d(channels, channels, 1, bias=False)
        else:
            self.noise_injector = None

        if min(resolution[0], resolution[1]) > 16:
            self.downscale = nn.Conv2d(channels, channels * 2, 3, padding=1, bias=False)
            self.inner = UNET(channels * 2, (resolution[0] // 2, resolution[1] // 2), inject_noise)
            self.upscale = nn.Conv2d(channels * 2, channels, 3, padding=1, bias=False)
        else:
            self.downscale = None
            self.inner = None
            self.upscale = None

    def forward(self, z):
        if self.noise_injector != None:
            noise = self.noise_injector(pink_noise(z.shape))
            z = z + noise

        z = self.block1(z)

        if self.downscale != None and self.inner != None and self.upscale != None:
            orig = z

            z = self.downscale(z)
            z = nn.functional.interpolate(z, scale_factor=0.5, mode="bilinear")
            z = self.inner(z)
            z = nn.functional.interpolate(z, size=orig.shape[2:], mode="bilinear")
            z = self.upscale(z)

            z = z + orig

        z = self.block2(z)

        return z

class MeanEstimator(nn.Module):
    def __init__(self):
        super(MeanEstimator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(2, mean_stdev_latent_dim, 1, bias=False),
            UNET(mean_stdev_latent_dim, resolution),
            nn.Conv2d(mean_stdev_latent_dim, 1, 1, bias=False),
        )

    def forward(self, original, mask):
        original = original.masked_fill(mask, 0)
        norm = abs_norm(original)

        inpainted = self.model(torch.cat([original / norm, mask], dim = 1))
        inpainted = inpainted * norm

        return torch.where(mask, inpainted, original)

class VarEstimator(nn.Module):
    def __init__(self):
        super(VarEstimator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(2, mean_stdev_latent_dim, 1, bias=False),
            UNET(mean_stdev_latent_dim, resolution),
            nn.Conv2d(mean_stdev_latent_dim, 1, 1, bias=False),
        )

    def forward(self, mean, mask):
        norm = abs_norm(mean)

        output = self.model(torch.cat([mean / norm, mask], dim = 1))
        output = output.abs() + 1e-9
        output = output * norm

        return output.masked_fill(~mask, 0)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, latent_dim, 1, bias=False),
            UNET(latent_dim, resolution, inject_noise=True),
            nn.Conv2d(latent_dim, 1, 1, bias=False),
        )

    def forward(self, mean, stdev, mask):
        norm = abs_norm(mean)

        inpainted = self.model(torch.cat([mean / norm, stdev / norm, mask], dim = 1))

        return torch.where(mask, inpainted, mean)
    
resolution = (128, 128)
latent_dim = 24
mean_stdev_latent_dim = 32
epoch = 0

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

G = CombinedGenerator("trained_128x128_b54a594.ckpt").to(device)
#D = Discriminator(latent_dim).to(device)

'''
hparams = {
    "G lr 0": 1e-3,
    "G lr 1": 1e-4,
    "D lr 0": 2e-3,
    "D lr 1": 2e-4,
    "G beta2 0": 0.9,
    "G beta2 1": 0.99,
    "D beta2 0": 0.9,
    "D beta2 1": 0.99,
    "GP Gamma 0": 1.0,
    "GP Gamma 1": 0.1,
    "Warmup": 5000,
    "Batch size": batch_size,
    "G params": sum(p.numel() for p in G.parameters()),
    "D params": sum(p.numel() for p in D.parameters()),
}

checkpoint = input("Enter the path to the checkpoint file (or press Enter to skip): ").strip()

if checkpoint:
    checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
    G.load_state_dict(checkpoint["generator"])
    D.load_state_dict(checkpoint["discriminator"])
    epoch = checkpoint["epoch"]
    print(f"Checkpoint loaded successfully at epoch {epoch}.")
'''

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
    rect_size = (64, 64)  # (width, height)
    top, left, bottom, right = center_of_mass_and_rectangle(mask, rect_size)

    with torch.no_grad():
        for i in range(masked_volume.shape[2]):
            slice_input = masked_volume[:, :, i]
            #slice_mask = mask[:, :, i]

            if not np.any(mask):
                continue  # Skip if no masked region

            cropped_input = slice_input[top:bottom, left:right]
            cropped_mask = mask[top:bottom, left:right]

            # Convert to tensors
            input_tensor = torch.tensor(cropped_input, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            mask_tensor = torch.tensor(cropped_mask, dtype=torch.bool).unsqueeze(0).unsqueeze(0)

            # Run model
            output = model(input_tensor, mask_tensor).squeeze().numpy()

            # Infill only masked region
            infilled_volume[top:bottom, left:right, i][cropped_mask] = output[cropped_mask]

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


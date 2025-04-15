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

def fft(image):
    ffted = torch.fft.fft2(image)
    return torch.cat([ffted.real, ffted.imag], dim=1)

def ifft(image):
    channels = image.shape[1] // 2
    real = image[:, :channels]
    imag = image[:, channels:]
    complex_tensor = torch.complex(real, imag)
    return torch.fft.ifft2(complex_tensor).real  # Only return the real part

def leaky_relu(z):
    return nn.functional.leaky_relu(z, 0.2)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels * 2, channels * 2, 3, groups=max(channels // 8, 1), padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels * 2, channels, 1, bias=False),
        )

    def forward(self, z):
        return z + self.block(z)

def extract_patches(x, patch_size):
    """
    Extract non-overlapping patches from x.

    Args:
        x (Tensor): Input tensor of shape (B, C, H, W)
        patch_size (int): Size of each (square) patch.

    Returns:
        Tensor: Extracted patches with shape 
                (B, C * patch_size * patch_size, H // patch_size, W // patch_size)
    """

    if type(patch_size) == type(0):
        patch_size = (patch_size, patch_size)

    # Create an unfold module with kernel_size and stride equal to patch_size.
    unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
    
    # Unfold the input; shape becomes (B, C * patch_size * patch_size, L)
    # where L = (H // patch_size) * (W // patch_size)
    patches = unfold(x)
    
    # Reshape to (B, C * patch_size * patch_size, H // patch_size, W // patch_size)
    H_patches = x.shape[2] // patch_size[0]
    W_patches = x.shape[3] // patch_size[1]
    patches = patches.view(x.shape[0], -1, H_patches, W_patches)
    
    return patches

def reconstruct_from_patches(patches, patch_size):
    """
    Reconstruct the original image from its patches.

    Args:
        patches (Tensor): Tensor of shape (B, C * patch_size * patch_size, H_patches, W_patches)
        patch_size (int): The same patch size used for extraction.

    Returns:
        Tensor: Reconstructed image of shape (B, C, H_patches * patch_size, W_patches * patch_size)
    """
    if type(patch_size) == type(0):
        patch_size = (patch_size, patch_size)

    B, patch_dim, H_patches, W_patches = patches.shape
    # Reshape patches to (B, patch_dim, L) with L = H_patches * W_patches.
    patches = patches.view(B, patch_dim, -1)
    
    # Define output size
    output_size = (H_patches * patch_size[0], W_patches * patch_size[1])
    
    # Create a fold module matching the patch parameters.
    fold = nn.Fold(output_size=output_size, kernel_size=patch_size, stride=patch_size)
    
    # For non-overlapping patches, each pixel is covered exactly once, so fold directly recovers the image.
    reconstructed = fold(patches)
    
    return reconstructed

class FFPatches(nn.Module):
    def __init__(self, channels, n_patches):
        super(FFPatches, self).__init__()
        self.n_patches = n_patches

        total_patches = n_patches[0] * n_patches[1]

        inner_channels = 2 * channels * total_patches

        self.block = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels * 2, 1, groups=total_patches),
            nn.LeakyReLU(0.2),
            nn.Conv2d(inner_channels * 2, inner_channels * 2, 3, groups=max(inner_channels // 8, 1), padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(inner_channels * 2, inner_channels, 1, groups=total_patches, bias=False),
        )

    def forward(self, z):
        orig = z

        z = extract_patches(z, self.n_patches)
        z = fft(z)
        z = self.block(z)
        z = ifft(z)
        z = reconstruct_from_patches(z, self.n_patches)

        z = match_shape(z, orig) # fft reshape
        z = orig + z
        orig_max = orig.amax(2, keepdim=True).amax(3, keepdim=True)
        orig_min = orig.amin(2, keepdim=True).amin(3, keepdim=True)
        z = z.clamp(orig_min, orig_max)

        return z

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(4, latent_dim, 1, bias=False),
            ResidualBlock(latent_dim),
            FFPatches(latent_dim, (1, 1)),
            ResidualBlock(latent_dim),
            FFPatches(latent_dim, (2, 2)),
            ResidualBlock(latent_dim),
            FFPatches(latent_dim, (4, 4)),
            ResidualBlock(latent_dim),
            FFPatches(latent_dim, (8, 1)),
            ResidualBlock(latent_dim),
            FFPatches(latent_dim, (1, 8)),
            ResidualBlock(latent_dim),
            FFPatches(latent_dim, (4, 4)),
            ResidualBlock(latent_dim),
            FFPatches(latent_dim, (2, 2)),
            ResidualBlock(latent_dim),
            FFPatches(latent_dim, (1, 1)),
            ResidualBlock(latent_dim),
            nn.Conv2d(latent_dim, 1, 1, bias=False),
        )

    def forward(self, original, mask):
        original = original.masked_fill(mask, 0)
        noise = torch.randn_like(original)
        fft_noise = ifft(torch.randn_like(fft(original)))
        inpainted = self.model(torch.cat([original, mask, noise, fft_noise], dim = 1))
        return torch.where(mask, inpainted, original)

resolution = (64, 64)
batch_size = 256
latent_dim = 8
epoch = 0

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

G = Generator(latent_dim).to(device)
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

def load_checkpoint():

    checkpoint = "trained_64x64_efa9a33_div8.ckpt" #input("Enter the path to the checkpoint file: ").strip()
    if checkpoint:
        checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
        G.load_state_dict(checkpoint["generator"])
        #D.load_state_dict(checkpoint["discriminator"])
        epoch = checkpoint["epoch"]
        print(f"Checkpoint loaded successfully at epoch {epoch}.")
        return checkpoint

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
    checkpoint = load_checkpoint()
    if checkpoint:
        model = G  # Generator already loaded

        with open("SegActi-45x201x201x614.bin", "rb") as f:
            w, x, y, z = 45, 201, 201, 614
            data = np.frombuffer(f.read(w * x * y * z * 4), dtype="f4").reshape(w, x, y, z)

        volume = draw_mask.choose_volume(data)
        masked_volume, mask = draw_mask.apply_mask(volume)
        #print("mask applied.")
        infill_and_display(model, masked_volume, mask)


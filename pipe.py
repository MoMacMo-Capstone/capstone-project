import torch
import torch.nn as nn
import torch.optim as optim
# from torch.utils.data import DataLoader
#from torch.utils.tensorboard import SummaryWriter
import torchvision
# from torchvision import datasets, transforms
from torchmetrics.image.fid import FrechetInceptionDistance

import lama_mask
#import read_seismic_data
from model import *

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector    # drawing
from matplotlib.path import Path                # drawing
import draw_mask

def zero_centered_gradient_penalty(Samples, Critics):
    Gradient, = torch.autograd.grad(outputs=Critics.sum(), inputs=Samples, create_graph=True)
    return Gradient.square().sum([1, 2, 3]).mean()

def generator_loss(discriminator, fake_noise, real_noise, mean, stdev, mask):
    fake_logits = discriminator(fake_noise, mean, stdev, mask)
    real_logits = discriminator(real_noise, mean, stdev, mask).detach()

    relativistic_logits = fake_logits - real_logits
    adversarial_loss = nn.functional.softplus(-relativistic_logits).mean()

    writer.add_scalar("Loss/Generator Loss", adversarial_loss.item(), step)

    return adversarial_loss

def backward_discriminator_loss(discriminator, fake_noise, real_noise, mean, stdev, mask, gamma):
    fake_noise = fake_noise.detach().requires_grad_(True)
    real_noise = real_noise.detach().requires_grad_(True)

    fake_logits = discriminator(fake_noise, mean, stdev, mask)
    real_logits = discriminator(real_noise, mean, stdev, mask)

    relativistic_logits = real_logits - fake_logits
    adversarial_loss = nn.functional.softplus(-relativistic_logits).mean()
    adversarial_loss.backward(retain_graph=True)

    r1_penalty = zero_centered_gradient_penalty(real_noise, real_logits)
    (r1_penalty * (gamma / 2)).backward()
    r2_penalty = zero_centered_gradient_penalty(fake_noise, fake_logits)
    (r2_penalty * (gamma / 2)).backward()

    writer.add_scalar("Loss/Discriminator Loss", adversarial_loss.item(), step)
    writer.add_scalar("Loss/R1 Penalty", r1_penalty.item(), step)
    writer.add_scalar("Loss/R2 Penalty", r2_penalty.item(), step)

    discriminator_loss = adversarial_loss.item() + (gamma / 2) * (r1_penalty.item() + r2_penalty.item())
    return discriminator_loss

def prepare_for_fid(imgs):
    imgs = imgs[:, 0:1]
    return (color_images(imgs) * 128).clamp(0, 255).to(torch.uint8)

def interpolate(x, x0, x1, y0, y1):
    if x <= x0:
        return y0
    elif x >= x1:
        return y1

    return (x - x0) * (y1 - y0) / (x1 - x0) + y0

def interpolate_exponential(x, x0, x1, y0, y1):
    if x <= x0:
        return y0
    elif x >= x1:
        return y1

    return y0 * (y1 / y0) ** ((x - x0) / (x1 - x0))

Mean = MeanEstimator().to(device)
Stdev = VarEstimator().to(device)
G = Generator().to(device)
D = Discriminator().to(device)

resolution = (64, 64)
latent_dim = 64
mean_stdev_latent_dim = 64
batch_size = 256
epoch = 0

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

G = Generator().to(device)
D = Discriminator().to(device)

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

def load_checkpoint():

    checkpoint = input("Enter the path to the checkpoint file: ").strip()
    if checkpoint:
        checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
        Mean.load_state_dict(checkpoint["mean"])
        Stdev.load_state_dict(checkpoint["stdev"])
        if "generator" in checkpoint and "discriminator" in checkpoint:
            G.load_state_dict(checkpoint["generator"])
            D.load_state_dict(checkpoint["discriminator"])
            step = checkpoint["step"]
        return checkpoint
    
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
            #stdev = torch.std(mask_tensor).item() if mask_tensor.numel() > 0 else 1.0

            stdev_val = torch.std(input_tensor[mask_tensor == 0]) if (mask_tensor == 0).any() else torch.tensor(1.0)
            stdev_tensor = torch.full_like(input_tensor, stdev_val)
            output = model(input_tensor, stdev_tensor, mask_tensor).squeeze().cpu().numpy()
            #output = model(input_tensor, stdev, mask_tensor).squeeze().cpu().numpy()
            infilled_volume[:, :, i][slice_mask] = output[slice_mask]
    
    draw_mask.show_volume_with_slider(volume, mask, infilled_volume)
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


import torch
import torch.nn as nn
import torch.optim as optim
# from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision

import lama_mask
import read_seismic_data
from model import *

batch_size = 256

Mean = MeanEstimator().to(device)
Stdev = VarEstimator().to(device)

print("Mean params:", sum(p.numel() for p in Mean.parameters()))
print("Stdev params:", sum(p.numel() for p in Stdev.parameters()))

writer = SummaryWriter()

hparams = {
    "lr": 1e-5,
    "beta1": 0.9,
    "beta2": 0.99,
    "Steps": 10000,
    "Batch size": batch_size,
    "Mean params": sum(p.numel() for p in Mean.parameters()),
    "Stdev params": sum(p.numel() for p in Stdev.parameters()),
}
checkpoint = None

if checkpoint:
    checkpoint = torch.load(checkpoint)
    Mean.load_state_dict(checkpoint["mean"])
    Stdev.load_state_dict(checkpoint["stdev"])
    step = checkpoint["step"]
else:
    step = 0

for name, value in hparams.items():
    writer.add_scalar(f"hparams/{name}", value, 0)

optimizer_Mean = optim.AdamW(Mean.parameters(), lr=hparams["lr"], betas=(hparams["beta1"], hparams["beta2"]))
optimizer_Stdev = optim.AdamW(Stdev.parameters(), lr=hparams["lr"], betas=(hparams["beta1"], hparams["beta2"]))

while step < hparams["Steps"]:
    step += 1

    chunk = read_seismic_data.get_multilayer_chunks(batch_size, resolution[0], slice_layers)
    chunk = abs_normalize(torch.tensor(chunk, device=device))
    chunk = chunk.detach().requires_grad_(True)

    mask = lama_mask.make_seismic_masks(batch_size, resolution)
    mask = torch.tensor(mask, device=device)

    mean = Mean(chunk, mask)
    real_imgs = chunk[:, slice_layers // 2:slice_layers // 2 + 1, :, :]

    optimizer_Mean.zero_grad()
    Mean_Loss = (mean - real_imgs).square().mean()
    Mean_Loss.backward()
    optimizer_Mean.step()

    mean = mean.detach()
    stdev = Stdev(chunk, mean, mask)

    optimizer_Stdev.zero_grad()
    unbounded_noise = inpainting_to_noise_unbounded(real_imgs, mean, stdev, mask)
    noise = inpainting_to_noise(real_imgs, mean, stdev, mask)
    Stdev_Loss = (stdev.abs() - (mean - real_imgs).abs()).square().mean()
    Stdev_Loss.backward()
    optimizer_Stdev.step()

    writer.add_scalar("Loss/Mean loss", Mean_Loss, step)
    writer.add_scalar("Loss/Stdev loss", Stdev_Loss, step)
    writer.add_scalar("Metrics/Noise L1 Norm", unbounded_noise.abs().mean(), step)
    writer.add_scalar("Metrics/Noise L2 Norm", unbounded_noise.square().mean().sqrt(), step)

    if step % 10 == 0:
        grid = torch.cat([
            real_imgs[:32, 0:1],
            noise[:32, 0:1],
            mean[:32, 0:1],
        ], 0)
        grid = nn.functional.interpolate(grid, scale_factor=2, mode="nearest")
        grid = color_images(grid)
        grid = torchvision.utils.make_grid(grid, nrow=32)
        writer.add_image("Images", grid, step)

    print(f"Step {step}: Mean Loss: {Mean_Loss.item():.05}, Stdev Loss: {Stdev_Loss.item():.05}")

    if step % 200 == 0:
        torch.save({
            "mean": Mean.state_dict(),
            "stdev": Stdev.state_dict(),
            "step": step,
        }, f"m_std_{resolution[0]}x{resolution[1]}_{mean_stdev_latent_dim}c_{step}.ckpt")

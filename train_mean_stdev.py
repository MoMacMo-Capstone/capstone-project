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
    "lr": 1e-4,
    "beta1": 0.9,
    "beta2": 0.99,
    "Steps": 10000,
    "Batch size": batch_size,
    "Mean params": sum(p.numel() for p in Mean.parameters()),
    "Stdev params": sum(p.numel() for p in Stdev.parameters()),
}
checkpoint = "m_std_128x128_32c_5000.ckpt"

if checkpoint:
    checkpoint = torch.load(checkpoint)
    Mean.load_state_dict(checkpoint["mean"])
    Stdev.load_state_dict(checkpoint["stdev"])
    step = checkpoint["step"]
else:
    step = 0

for name, value in hparams.items():
    writer.add_scalar(f"hparams/{name}", value, 0)

optimizer_M_STD = optim.AdamW(list(Mean.parameters()) + list(Stdev.parameters()), lr=hparams["lr"], betas=(hparams["beta1"], hparams["beta2"]))

while step < hparams["Steps"]:
    step += 1

    real_imgs = read_seismic_data.get_chunks(batch_size, resolution[0])
    real_imgs = abs_normalize(torch.tensor(real_imgs, device=device))
    real_imgs = real_imgs.detach().requires_grad_(True)

    mask = lama_mask.make_seismic_masks(batch_size, resolution)
    mask = torch.tensor(mask, device=device)

    mean = Mean(real_imgs, mask)
    stdev = Stdev(mean.detach(), mask)

    optimizer_M_STD.zero_grad()
    M_Loss = (mean - real_imgs).square().mean()
    STD_Loss = (stdev.square() - (mean - real_imgs).square()).square().mean()
    (M_Loss + STD_Loss).backward()
    optimizer_M_STD.step()

    writer.add_scalar("Loss/Mean loss", M_Loss, step)
    writer.add_scalar("Loss/Stdev loss", STD_Loss, step)

    grid = torch.cat([
        real_imgs[:32, 0:1],
        stdev[:32, 0:1],
        mean[:32, 0:1],
    ], 0)
    # grid = nn.functional.interpolate(grid, scale_factor=2, mode="nearest")
    grid = color_images(grid)
    grid = torchvision.utils.make_grid(grid, nrow=32)
    writer.add_image("Images", grid, step)

    print(f"Step {step}: Mean Loss: {M_Loss.item():.05}, Stdev Loss: {STD_Loss.item():.05}")

    if step % 200 == 0:
        torch.save({
            "mean": Mean.state_dict(),
            "stdev": Stdev.state_dict(),
            "step": step,
        }, f"m_std_{resolution[0]}x{resolution[1]}_{mean_stdev_latent_dim}c_{step}.ckpt")

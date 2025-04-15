import torch
import torch.nn as nn
import torch.optim as optim
# from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
# from torchvision import datasets, transforms
from torchmetrics.image.fid import FrechetInceptionDistance

import lama_mask
import read_seismic_data
from model import *

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

batch_size = 256
step = 0

Mean = MeanEstimator().to(device)
Stdev = VarEstimator().to(device)
G = Generator().to(device)
D = Discriminator().to(device)

print("Mean params:", sum(p.numel() for p in Mean.parameters()))
print("Stdev params:", sum(p.numel() for p in Stdev.parameters()))
print("G params:", sum(p.numel() for p in G.parameters()))
print("D params:", sum(p.numel() for p in D.parameters()))
# exit()

writer = SummaryWriter()

hparams = {
    "G lr 0": 2.5e-5,
    "G lr 1": 5e-6,
    "D lr 0": 1e-4,
    "D lr 1": 2e-5,
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
checkpoint = "m_std_checkpoint_5000.ckpt"

if checkpoint:
    checkpoint = torch.load(checkpoint)
    Mean.load_state_dict(checkpoint["mean"])
    Stdev.load_state_dict(checkpoint["stdev"])
    if "generator" in checkpoint and "discriminator" in checkpoint:
        G.load_state_dict(checkpoint["generator"])
        D.load_state_dict(checkpoint["discriminator"])
        step = checkpoint["step"]

for name, value in hparams.items():
    writer.add_scalar(f"hparams/{name}", value, 0)

optimizer_G = optim.AdamW(G.parameters(), lr=hparams["G lr 0"], betas=(0.0, hparams["G beta2 0"]))
optimizer_D = optim.AdamW(D.parameters(), lr=hparams["D lr 0"], betas=(0.0, hparams["D beta2 0"]))

# transform = transforms.Compose([
    # transforms.ToTensor(),
    # transforms.Normalize((0.5,), (0.5,))
# ])
# train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
# train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
fid_metric_acc = FrechetInceptionDistance(feature = 2048).to(device)
fid_metric = FrechetInceptionDistance(feature = 2048).to(device)

# mask = torch.zeros((batch_size, 1, 28, 28), device=device, dtype=torch.bool)
# mask[:,:,:,9:21] = 1
# mask[:,:,9:21,9:21] = 1

while True:
    step += 1

    warmup = hparams["Warmup"]

    G_lr = interpolate_exponential(step, 1, warmup, hparams["G lr 0"], hparams["G lr 1"])
    D_lr = interpolate_exponential(step, 1, warmup, hparams["D lr 0"], hparams["D lr 1"])
    G_beta2 = 1 - interpolate_exponential(step, 1, warmup, 1 - hparams["G beta2 0"], 1 - hparams["G beta2 1"])
    D_beta2 = 1 - interpolate_exponential(step, 1, warmup, 1 - hparams["D beta2 0"], 1 - hparams["D beta2 1"])
    GP_gamma = interpolate_exponential(step, 1, warmup, hparams["GP Gamma 0"], hparams["GP Gamma 1"])

    for param_group in optimizer_G.param_groups:
        param_group['lr'] = G_lr
        param_group['betas'] = (0, G_beta2)

    for param_group in optimizer_D.param_groups:
        param_group['lr'] = D_lr
        param_group['betas'] = (0, D_beta2)

    real_imgs = read_seismic_data.get_chunks(batch_size, resolution[0])
    real_imgs = abs_normalize(torch.tensor(real_imgs, device=device))
    real_imgs = real_imgs.detach().requires_grad_(True)

    mask = lama_mask.make_seismic_masks(batch_size, resolution)
    mask = torch.tensor(mask, device=device)

    mean = Mean(real_imgs, mask).detach()
    stdev = Stdev(mean, mask).detach()
    fake_noise = G(mean.detach(), stdev.detach(), mask)

    fake_imgs = noise_to_inpainting(fake_noise, mean, stdev, mask)
    real_noise = inpainting_to_noise(real_imgs, mean, stdev, mask)

    writer.add_scalar("Metrics/L1 loss", (fake_imgs - real_imgs).abs().mean(), step)
    writer.add_scalar("Metrics/L2 loss", (fake_imgs - real_imgs).square().mean(), step)

    optimizer_G.zero_grad()
    G_loss = generator_loss(D, fake_noise, real_noise, mean, stdev, mask)
    G_loss.backward()
    optimizer_G.step()

    optimizer_D.zero_grad()
    D_loss = backward_discriminator_loss(D, fake_noise, real_noise, mean, stdev, mask, GP_gamma)
    optimizer_D.step()

    grid = torch.cat([
        real_imgs[:32, 0:1],
        stdev[:32, 0:1],
        mean[:32, 0:1],
        fake_imgs[:32, 0:1],
    ], 0)
    grid = nn.functional.interpolate(grid, scale_factor=2, mode="nearest")
    grid = color_images(grid)
    grid = torchvision.utils.make_grid(grid, nrow=32)
    writer.add_image("Images", grid, step)

    print(f"Step {step}: D Loss: {D_loss:.05}, G Loss: {G_loss.item():.05}")

    if step % 200 == 0:
        fid_metric_acc.update(prepare_for_fid(real_imgs), real = True)
        fid_metric.reset()
        fid_metric.merge_state(fid_metric_acc)
        fid_metric.update(prepare_for_fid(fake_imgs), real = False)
        fid_score = fid_metric.compute()
        print(f"FID: {fid_score}")
        writer.add_scalar("Metrics/FID", fid_score.item(), step)
        torch.save({
            "mean": Mean.state_dict(),
            "stdev": Stdev.state_dict(),
            "generator": G.state_dict(),
            "discriminator": D.state_dict(),
            "step": step,
        }, f"checkpoint_4_{step}.ckpt")

import math
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

def fft(image):
    image = torch.fft.rfft2(image)
    return torch.cat([image.real, image.imag], dim=1)

def ifft(image):
    channels = image.shape[1] // 2
    image = torch.complex(image[:,:channels], image[:,channels:])
    return torch.fft.irfft2(image)

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

def abs_norm(images):
    return images.abs().amax(2, keepdim=True).amax(3, keepdim=True) + 1e-9

def abs_normalize(images):
    return images / abs_norm(images)

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

class Generator(nn.Module):
    def __init__(self, latent_dim, resolution, inject_noise):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(2, latent_dim, 1, bias=False),
            UNET(latent_dim, resolution, inject_noise),
            nn.Conv2d(latent_dim, 1, 1, bias=False),
        )

    def forward(self, original, mask):
        original = original.masked_fill(mask, 0)
        norm = abs_norm(original)

        original = original / norm
        inpainted = self.model(torch.cat([original, mask], dim = 1))
        inpainted = inpainted * norm

        return torch.where(mask, inpainted, original)

class Discriminator(nn.Module):
    def __init__(self, latent_dim, resolution):
        super(Discriminator, self).__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(3, latent_dim, 1, bias=False),
            UNET(latent_dim, resolution),
        )
        self.linear = nn.Linear(latent_dim, 1)

    def forward(self, inpainted, mask):
        z = torch.cat([inpainted, mask], dim=1)
        z = self.convs(z).mean((2, 3))
        return self.linear(z)

def zero_centered_gradient_penalty(Samples, Critics):
    Gradient, = torch.autograd.grad(outputs=Critics.sum(), inputs=Samples, create_graph=True)
    return Gradient.square().sum([1, 2, 3]).mean()

def masked_zero_centered_gradient_penalty(samples, critics, mask):
    grad, = torch.autograd.grad(outputs=critics.sum(), inputs=samples, create_graph=True)
    grad *= mask
    return grad.square().sum([1, 2, 3]).mean()

def generator_loss(discriminator, fake_samples, real_samples, mask):
    fake_logits = discriminator(fake_samples, mask)
    real_logits = discriminator(real_samples, mask)

    relativistic_logits = fake_logits - real_logits
    adversarial_loss = nn.functional.softplus(-relativistic_logits).mean()

    writer.add_scalar("Loss/Generator Loss", adversarial_loss.item(), epoch)

    return adversarial_loss

def backward_discriminator_loss(discriminator, fake_samples, real_samples, mask, gamma):
    fake_samples = fake_samples.detach().requires_grad_(True)
    real_samples = real_samples.detach().requires_grad_(True)

    fake_logits = discriminator(fake_samples, mask)
    real_logits = discriminator(real_samples, mask)

    relativistic_logits = real_logits - fake_logits
    adversarial_loss = nn.functional.softplus(-relativistic_logits).mean()
    adversarial_loss.backward(retain_graph=True)

    # r1_penalty = masked_zero_centered_gradient_penalty(real_samples, real_logits, mask)
    # r2_penalty = masked_zero_centered_gradient_penalty(fake_samples, fake_logits, mask)
    r1_penalty = zero_centered_gradient_penalty(real_samples, real_logits)
    (r1_penalty * (gamma / 2)).backward()
    r2_penalty = zero_centered_gradient_penalty(fake_samples, fake_logits)
    (r2_penalty * (gamma / 2)).backward()

    writer.add_scalar("Loss/Discriminator Loss", adversarial_loss.item(), epoch)
    writer.add_scalar("Loss/R1 Penalty", r1_penalty.item(), epoch)
    writer.add_scalar("Loss/R2 Penalty", r2_penalty.item(), epoch)

    discriminator_loss = adversarial_loss.item() + (gamma / 2) * (r1_penalty.item() + r2_penalty.item())
    return discriminator_loss

def color_images(images):
    images = abs_normalize(images) * 2
    return torch.cat([images, images.abs() - 1, -images], dim=1).clamp(0, 1)

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

resolution = (32, 32)
batch_size = 256
latent_dim = 8
epoch = 0

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

G = Generator(latent_dim, resolution, inject_noise=True).to(device)
D = Discriminator(latent_dim, resolution).to(device)

print("G params:", sum(p.numel() for p in G.parameters()))
print("D params:", sum(p.numel() for p in D.parameters()))
# exit()

writer = SummaryWriter()

hparams = {
    "G lr 0": 1e-4,
    "G lr 1": 1e-5,
    "D lr 0": 5e-4,
    "D lr 1": 5e-5,
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
checkpoint = None

if checkpoint:
    checkpoint = torch.load(checkpoint)
    G.load_state_dict(checkpoint["generator"])
    D.load_state_dict(checkpoint["discriminator"])
    epoch = checkpoint["epoch"]

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
    epoch += 1

    warmup = hparams["Warmup"]

    G_lr = interpolate_exponential(epoch, 1, warmup, hparams["G lr 0"], hparams["G lr 1"])
    D_lr = interpolate_exponential(epoch, 1, warmup, hparams["D lr 0"], hparams["D lr 1"])
    G_beta2 = 1 - interpolate_exponential(epoch, 1, warmup, 1 - hparams["G beta2 0"], 1 - hparams["G beta2 1"])
    D_beta2 = 1 - interpolate_exponential(epoch, 1, warmup, 1 - hparams["D beta2 0"], 1 - hparams["D beta2 1"])
    GP_gamma = interpolate_exponential(epoch, 1, warmup, hparams["GP Gamma 0"], hparams["GP Gamma 1"])

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

    # imgs1_stage1 = G1(real_imgs, mask)
    # imgs2_stage1 = G1(real_imgs, ~mask)

    # optimizer_G1.zero_grad()
    # G1_loss = (imgs1_stage1 - imgs2_stage1).abs().mean()
    # G1_loss.backward()
    # optimizer_G1.step()

    # writer.add_scalar("Loss/L1 loss stage 1", G1_loss, epoch)

    # imgs1_stage2 = G(imgs1_stage1.detach(), mask)
    # imgs2_stage2 = G(imgs2_stage1.detach(), ~mask)

    fake_imgs = G(real_imgs, mask)
    real_imgs, fake_imgs = torch.cat([real_imgs, fake_imgs], dim=1), torch.cat([fake_imgs, real_imgs], dim=1)

    writer.add_scalar("Metrics/L1 loss", (fake_imgs - real_imgs).abs().mean(), epoch)

    optimizer_G.zero_grad()
    G_loss = generator_loss(D, fake_imgs, real_imgs, mask)
    G_loss.backward()
    optimizer_G.step()

    optimizer_D.zero_grad()
    D_loss = backward_discriminator_loss(D, fake_imgs, real_imgs, mask, GP_gamma)
    optimizer_D.step()

    grid = torch.cat([
        real_imgs[:32, 0:1],
        real_imgs[:32, 0:1].masked_fill(mask[:32, 0:1], 0),
        fake_imgs[:32, 0:1],
    ], 0)
    grid = nn.functional.interpolate(grid, scale_factor=2, mode="nearest")
    grid = color_images(grid)
    grid = torchvision.utils.make_grid(grid, nrow=32)
    writer.add_image("Images", grid, epoch)

    print(f"Epoch {epoch}: D Loss: {D_loss:.05}, G Loss: {G_loss.item():.05}")

    if epoch % 200 == 0:
        fid_metric_acc.update(prepare_for_fid(real_imgs), real = True)
        fid_metric.reset()
        fid_metric.merge_state(fid_metric_acc)
        fid_metric.update(prepare_for_fid(fake_imgs), real = False)
        fid_score = fid_metric.compute()
        print(f"FID: {fid_score}")
        writer.add_scalar("Metrics/FID", fid_score.item(), epoch)
        torch.save({
            "generator": G.state_dict(),
            "discriminator": D.state_dict(),
            "epoch": epoch,
        }, f"checkpoint_3_{epoch}.ckpt")

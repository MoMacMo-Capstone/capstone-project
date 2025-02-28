import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import datasets, transforms
from torchmetrics.image.fid import FrechetInceptionDistance

# import numpy as np
# import mask_functions

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels * 2, channels * 2, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels * 2, channels, 1, bias=False),
        )

    def forward(self, z):
        return z + self.block(z)

class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DBlock, self).__init__()
        self.block = nn.Sequential(
            ResidualBlock(in_channels),
            ResidualBlock(in_channels),
            ResidualBlock(in_channels),
            ResidualBlock(in_channels),
        )

        if in_channels != out_channels:
            self.block.append(nn.Conv2d(in_channels, out_channels, 3, bias=False, padding = 1))

    def forward(self, z):
        return nn.functional.interpolate(self.block(z), scale_factor=0.5, mode="bilinear")

class GBlock(nn.Module):
    def __init__(self, channels, resolution):
        super(GBlock, self).__init__()
        self.input = nn.Sequential(ResidualBlock(channels), ResidualBlock(channels))
        self.output = nn.Sequential(ResidualBlock(channels), ResidualBlock(channels))
        self.resolution = resolution

        self.inner = None
        self.noise_injector = None
        self.channel_up = None
        self.channel_down = None

        if resolution[0] // 2 >= 1 and resolution[1] // 2 >= 1:
            self.inner = GBlock(channels * 2, (resolution[0] // 2, resolution[1] // 2))
            self.noise_injector = nn.Conv2d(channels * 4, channels * 2, 1, bias=False)
            self.channel_up = nn.Conv2d(channels, channels * 2, 3, bias=False, padding=1)
            self.channel_down = nn.Conv2d(channels * 2, channels, 3, bias=False, padding=1)

    def forward(self, x):
        x = self.input(x)

        if self.inner and self.channel_up and self.channel_down and self.noise_injector:
            x_orig = x

            x = self.channel_up(x)
            x = nn.functional.interpolate(x, scale_factor=0.5, mode="bilinear")

            x = self.noise_injector(torch.cat([x, torch.randn_like(x)], 1))
            x = self.inner(x)

            x = nn.functional.interpolate(x, size=self.resolution, mode="bilinear")
            x = self.channel_down(x)

            x += x_orig

        return self.output(x)

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(2, latent_dim, 1, bias=False),
            GBlock(latent_dim, (28, 28)),
            nn.Conv2d(latent_dim, 1, 1, bias=False),
        )

    def forward(self, original, mask):
        original = original.masked_fill(mask, 0)
        inpainting = self.model(torch.cat([original, mask], dim = 1))
        return torch.where(mask, inpainting, original)

class Discriminator(nn.Module):
    def __init__(self, latent_dim):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(2, latent_dim, 1),
            DBlock(latent_dim, latent_dim * 2),
            DBlock(latent_dim * 2, latent_dim * 4),
            DBlock(latent_dim * 4, latent_dim * 8),
            DBlock(latent_dim * 8, latent_dim * 16),
            nn.Flatten(),
            nn.Linear(latent_dim * 16, 1),
        )

    def forward(self, inpainted_pair, mask):
        z = torch.cat([inpainted_pair, mask], dim = 1);
        return self.model(z)

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

def discriminator_loss(discriminator, fake_samples, real_samples, mask, gamma):
    fake_samples = fake_samples.detach().requires_grad_(True)
    real_samples = real_samples.detach().requires_grad_(True)

    fake_logits = discriminator(fake_samples, mask)
    real_logits = discriminator(real_samples, mask)

    r1_penalty = masked_zero_centered_gradient_penalty(real_samples, real_logits, mask)
    r2_penalty = masked_zero_centered_gradient_penalty(fake_samples, fake_logits, mask)

    # r1_penalty = zero_centered_gradient_penalty(real_samples, real_logits)
    # r2_penalty = zero_centered_gradient_penalty(fake_samples, fake_logits)

    relativistic_logits = real_logits - fake_logits
    adversarial_loss = nn.functional.softplus(-relativistic_logits).mean()

    writer.add_scalar("Loss/Discriminator Loss", adversarial_loss.item(), epoch)
    writer.add_scalar("Loss/R1 Penalty", r1_penalty.item(), epoch)
    writer.add_scalar("Loss/R2 Penalty", r2_penalty.item(), epoch)

    discriminator_loss = adversarial_loss + (gamma / 2) * (r1_penalty + r2_penalty)
    return discriminator_loss

def color_images(images):
    return torch.cat([images.relu(), torch.zeros_like(images), (-images).relu()], dim=1)

def interpolate(epoch, epochs, start, end):
    a = max(min(epoch / (epochs - 1), 1), 0)
    return start * (1 - a) + end * a

def interpolate_exponential(x, x0, x1, y0, y1):
    if x <= x0:
        return y0
    elif x >= x1:
        return y1

    return y0 * (y1 / y0) ** ((x - x0) / (x1 - x0))

def prepare_for_fid(imgs):
    imgs = imgs[:, 0:1]
    imgs = imgs * 128 + 128
    imgs = imgs.clamp(0, 255).to(torch.uint8)
    imgs = imgs.broadcast_to((imgs.shape[0], 3, imgs.shape[2], imgs.shape[3]))
    return imgs

batch_size = 256
latent_dim = 8

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

G = Generator(latent_dim).to(device)
D = Discriminator(latent_dim).to(device)

optimizer_G = optim.AdamW(G.parameters(), lr=2e-4, betas=(0.0, 0.9))
optimizer_D = optim.AdamW(D.parameters(), lr=2e-4, betas=(0.0, 0.9))
loss_function = nn.BCELoss()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
fid_metric_acc = FrechetInceptionDistance(feature = 2048)
fid_metric = FrechetInceptionDistance(feature = 2048)
writer = SummaryWriter()

mask = torch.zeros((batch_size, 1, 28, 28), device=device, dtype=torch.bool)
mask[:,:,9:21,9:21] = 1

epoch = 0

while True:
    epoch += 1

    real_imgs = next(iter(train_loader))[0]

    # real_imgs = np.concatenate([np.expand_dims(mask_functions.get_chunk(28), (0, 1)) for _ in range(256)])
    # real_imgs = torch.tensor(real_imgs)

    real_imgs = real_imgs.to(device).detach().requires_grad_(True)
    fake_imgs = G(real_imgs, mask)

    writer.add_scalar("Metrics/L1 loss", (fake_imgs - real_imgs).abs().mean(), epoch)
    writer.add_scalar("Metrics/L2 loss", (fake_imgs - real_imgs).square().mean(), epoch)

    # real_imgs, fake_imgs = torch.cat([real_imgs, fake_imgs], dim=1), torch.cat([fake_imgs, real_imgs], dim=1)

    optimizer_D.zero_grad()
    D_loss = discriminator_loss(D, fake_imgs, real_imgs, mask, 1.0)
    D_loss.backward()
    optimizer_D.step()

    optimizer_G.zero_grad()
    G_loss = generator_loss(D, fake_imgs, real_imgs, mask)
    G_loss.backward()
    optimizer_G.step()

    grid = torch.cat([
        real_imgs[:32, 0:1],
        real_imgs[:32, 0:1].masked_fill(mask[:32, 0:1], 0),
        fake_imgs[:32, 0:1]
    ], 0)
    grid = nn.functional.interpolate(grid, scale_factor=2, mode="nearest")
    grid = color_images(grid)
    grid = torchvision.utils.make_grid(grid, nrow=32)
    writer.add_image("Images", grid, epoch)

    print(f"Epoch {epoch}: D Loss: {D_loss.item()}, G Loss: {G_loss.item()}")

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
        }, f"checkpoint_{epoch}.bin")

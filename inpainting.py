import torch
import torch.nn as nn
import torch.optim as optim
# from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
# from torchvision import datasets, transforms
# from torchmetrics.image.fid import FrechetInceptionDistance

import lama_mask
import read_seismic_data

def fft(image):
    image = torch.fft.rfft2(image)
    return torch.cat([image.real, image.imag], dim=1)

def ifft(image):
    channels = image.shape[1] // 2
    image = torch.complex(image[:,:channels], image[:,channels:])
    return torch.fft.irfft2(image)

def leaky_relu(z):
    return nn.functional.leaky_relu(z, 0.2)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels * 2, channels * 2, 3, groups=channels // 8, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels * 2, channels, 1, bias=False),
        )

    def forward(self, z):
        return z + self.block(z)

class FFResidualBlock(nn.Module):
    def __init__(self, channels, pe_channels, resolution):
        super(FFResidualBlock, self).__init__()
        self.pe = torch.randn((pe_channels, resolution[0], resolution[1] // 2 + 1))
        self.inner = nn.Sequential(
            nn.Conv2d(channels * 2 + pe_channels, channels * 4, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels * 4, channels * 4, 3, groups=channels // 4, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels * 4, channels * 2, 1, bias=False),
        )

    def forward(self, z):
        orig = z

        z = fft(z)
        pe = self.pe.expand(z.shape[0], self.pe.shape[0], self.pe.shape[1], self.pe.shape[2])
        z = self.inner(torch.cat([z, pe], dim=1))
        z = ifft(z)

        z = z - z.mean([2, 3], keepdim=True) + orig.mean([2, 3], keepdim=True)
        orig_max = orig.amax(2, keepdim=True).amax(3, keepdim=True)
        orig_min = orig.amin(2, keepdim=True).amin(3, keepdim=True)
        z = z.clamp(orig_min, orig_max)

        return z + orig

class Generator(nn.Module):
    def __init__(self, latent_dim, resolution):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(4, latent_dim, 1, bias=False),
            FFResidualBlock(latent_dim, 8, resolution),
            ResidualBlock(latent_dim),
            FFResidualBlock(latent_dim, 8, resolution),
            ResidualBlock(latent_dim),
            FFResidualBlock(latent_dim, 8, resolution),
            ResidualBlock(latent_dim),
            FFResidualBlock(latent_dim, 8, resolution),
            ResidualBlock(latent_dim),
            nn.Conv2d(latent_dim, 1, 1, bias=False),
        )

    def forward(self, original, mask):
        original = original.masked_fill(mask, 0)
        noise = torch.randn_like(original)
        fft_noise = ifft(torch.randn_like(fft(original)))
        inpainted = self.model(torch.cat([original, mask, noise, fft_noise], dim = 1))
        return torch.where(mask, inpainted, original)

class Discriminator(nn.Module):
    def __init__(self, latent_dim, resolution):
        super(Discriminator, self).__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(3, latent_dim, 1),
            FFResidualBlock(latent_dim, 8, resolution),
            ResidualBlock(latent_dim),
            FFResidualBlock(latent_dim, 8, resolution),
            ResidualBlock(latent_dim),
            FFResidualBlock(latent_dim, 8, resolution),
            ResidualBlock(latent_dim),
            FFResidualBlock(latent_dim, 8, resolution),
            ResidualBlock(latent_dim),
        )
        self.linear = nn.Linear(latent_dim, 1)

    def forward(self, inpainted, mask):
        z = torch.cat([inpainted, mask], dim = 1);
        z = self.convs(z).mean([2, 3])
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

def discriminator_loss(discriminator, fake_samples, real_samples, mask, gamma):
    fake_samples = fake_samples.detach().requires_grad_(True)
    real_samples = real_samples.detach().requires_grad_(True)

    fake_logits = discriminator(fake_samples, mask)
    real_logits = discriminator(real_samples, mask)

    # r1_penalty = masked_zero_centered_gradient_penalty(real_samples, real_logits, mask)
    # r2_penalty = masked_zero_centered_gradient_penalty(fake_samples, fake_logits, mask)
    r1_penalty = zero_centered_gradient_penalty(real_samples, real_logits)
    r2_penalty = zero_centered_gradient_penalty(fake_samples, fake_logits)

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

resolution = (64, 64)
batch_size = 256
latent_dim = 32

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

G = Generator(latent_dim, resolution).to(device)
D = Discriminator(latent_dim, resolution).to(device)

writer = SummaryWriter()

hparams = {
    "G lr": 5e-4,
    "D lr": 1e-3,
    "G beta2": 0.9,
    "D beta2": 0.9,
    "GP Gamma": 1.0,
}

for name, value in hparams.items():
    writer.add_scalar(f"hparams/{name}", value, 0)

optimizer_G = optim.AdamW(G.parameters(), lr=hparams["G lr"], betas=(0.0, hparams["G beta2"]))
optimizer_D = optim.AdamW(D.parameters(), lr=hparams["D lr"], betas=(0.0, hparams["D beta2"]))
loss_function = nn.BCELoss()

# transform = transforms.Compose([
    # transforms.ToTensor(),
    # transforms.Normalize((0.5,), (0.5,))
# ])
# train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
# train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
# fid_metric_acc = FrechetInceptionDistance(feature = 2048)
# fid_metric = FrechetInceptionDistance(feature = 2048)

# mask = torch.zeros((batch_size, 1, 28, 28), device=device, dtype=torch.bool)
# mask[:,:,:,9:21] = 1
# mask[:,:,9:21,9:21] = 1

epoch = 0

while True:
    epoch += 1

    real_imgs = read_seismic_data.get_chunks(batch_size, resolution[0])
    real_imgs = torch.tensor(real_imgs)

    mask = lama_mask.make_seismic_masks(batch_size, resolution)
    mask = torch.tensor(mask)

    real_imgs = real_imgs.to(device).detach().requires_grad_(True)
    fake_imgs = G(real_imgs, mask)

    writer.add_scalar("Metrics/L1 loss", (fake_imgs - real_imgs).abs().mean(), epoch)
    writer.add_scalar("Metrics/L2 loss", (fake_imgs - real_imgs).square().mean(), epoch)

    real_imgs, fake_imgs = torch.cat([real_imgs, fake_imgs], dim=1), torch.cat([fake_imgs, real_imgs], dim=1)

    optimizer_G.zero_grad()
    G_loss = generator_loss(D, fake_imgs, real_imgs, mask)
    G_loss.backward()
    optimizer_G.step()

    optimizer_D.zero_grad()
    D_loss = discriminator_loss(D, fake_imgs, real_imgs, mask, hparams["GP Gamma"])
    D_loss.backward()
    optimizer_D.step()

    grid = torch.cat([
        real_imgs[:32, 0:1],
        real_imgs[:32, 0:1].masked_fill(mask[:32, 0:1], 0),
        fake_imgs[:32, 0:1]
    ], 0)
    # grid = nn.functional.interpolate(grid, scale_factor=2, mode="nearest")
    grid = color_images(grid)
    grid = torchvision.utils.make_grid(grid, nrow=32)
    writer.add_image("Images", grid, epoch)

    print(f"Epoch {epoch}: D Loss: {D_loss.item()}, G Loss: {G_loss.item()}")

    # if epoch % 200 == 0:
        # fid_metric_acc.update(prepare_for_fid(real_imgs), real = True)
        # fid_metric.reset()
        # fid_metric.merge_state(fid_metric_acc)
        # fid_metric.update(prepare_for_fid(fake_imgs), real = False)
        # fid_score = fid_metric.compute()
        # print(f"FID: {fid_score}")
        # writer.add_scalar("Metrics/FID", fid_score.item(), epoch)
        # torch.save({
            # "generator": G.state_dict(),
            # "discriminator": D.state_dict(),
            # "epoch": epoch,
        # }, f"checkpoint_{epoch}.ckpt")

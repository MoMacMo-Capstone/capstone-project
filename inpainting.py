import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import datasets, transforms
from torchmetrics.image.fid import FrechetInceptionDistance

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

    def forward(self, inpainted, mask):
        z = torch.cat([inpainted, mask], dim = 1);
        return self.model(z)

def zero_centered_gradient_penalty(Samples, Critics):
    Gradient, = torch.autograd.grad(outputs=Critics.sum(), inputs=Samples, create_graph=True)
    return Gradient.square().sum([1, 2, 3])

def masked_zero_centered_gradient_penalty(samples, critics, mask):
    grad, = torch.autograd.grad(outputs=critics.sum(), inputs=samples, create_graph=True)
    grad = grad.masked_fill(mask, 0)
    return grad.square().sum([1, 2, 3])

def generator_loss(discriminator, fake_samples, real_logits, mask):
    fake_logits = discriminator(fake_samples, mask)

    relativistic_logits = fake_logits - real_logits.detach()
    adversarial_loss = nn.functional.softplus(-relativistic_logits)

    return adversarial_loss.mean()

def discriminator_loss(discriminator, fake_samples, real_samples, real_logits, mask, gamma):
    fake_samples = fake_samples.detach().requires_grad_(True)

    fake_logits = discriminator(fake_samples, mask)

    r1_penalty = masked_zero_centered_gradient_penalty(real_samples, real_logits, ~mask)
    r2_penalty = masked_zero_centered_gradient_penalty(fake_samples, fake_logits, ~mask)

    relativistic_logits = real_logits - fake_logits
    adversarial_loss = nn.functional.softplus(-relativistic_logits)

    discriminator_loss = adversarial_loss + (gamma / 2) * (r1_penalty + r2_penalty)
    return discriminator_loss.mean()

def image_rows(imgs, num_images=10):
    np = [img.detach().cpu().numpy() for img in imgs]
    _, axes = plt.subplots(len(np), num_images, figsize=(num_images, len(np)))
    for i in range(len(np)):
        for j in range(num_images):
            axes[i][j].imshow(np[i][j, 0], cmap="seismic", norm=matplotlib.colors.NoNorm(-1, 1, clip=True))
            axes[i][j].axis("off")
    return plt

def save_image_rows(file, imgs, num_images=10):
    plt = image_rows(imgs, num_images)
    plt.savefig(file)
    plt.close()

def show_image_rows(imgs, num_images=10):
    plt = image_rows(imgs, num_images)
    plt.show()
    plt.close()

def show_images(imgs, num_images=10):
    show_image_rows([imgs], num_images)

def show_generated_images(generator, latent_dim, num_images=10):
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(num_images, latent_dim)
        fake_imgs = generator(noise).cpu().numpy()

    show_images(fake_imgs, num_images)
    generator.train()

epochs = 10000
batch_size = 256
latent_dim = 8

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

G = Generator(latent_dim).to(device)
D = Discriminator(latent_dim).to(device)

optimizer_G = optim.AdamW(G.parameters(), lr=5e-4, betas=(0.0, 0.9))
optimizer_D = optim.AdamW(D.parameters(), lr=5e-4, betas=(0.0, 0.9))
loss_function = nn.BCELoss()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
fid_metric = FrechetInceptionDistance(feature = 2048)
writer = SummaryWriter()

mask = torch.zeros((batch_size, 1, 28, 28), device=device).to(torch.bool)
mask[:,:,9:21,9:21] = 1

for epoch in range(1, epochs + 1):
    optimizer_D.zero_grad()

    real_imgs = next(iter(train_loader))[0]
    real_imgs = real_imgs.to(device).detach().requires_grad_(True)
    fake_imgs = G(real_imgs, mask)

    real_logits = D(real_imgs, mask)

    optimizer_G.zero_grad()
    adv_loss = generator_loss(D, fake_imgs, real_logits, mask)
    rec_loss = nn.functional.l1_loss(fake_imgs, real_imgs)
    G_loss = adv_loss + rec_loss * 100.
    G_loss.backward()
    optimizer_G.step()

    optimizer_D.zero_grad()
    D_loss = discriminator_loss(D, fake_imgs, real_imgs, real_logits, mask, 0.1)
    D_loss.backward()
    optimizer_D.step()

    if epoch % 10 == 0:
        # save_image_rows(f"fig_2_{epoch:05}.png", [real_imgs, real_imgs.masked_fill(mask, 0), fake_imgs])
        writer.add_images('Real Images', real_imgs, epoch)
        writer.add_images('Masked Images', real_imgs.masked_fill(mask, 0), epoch)
        writer.add_images('Fake Images', fake_imgs, epoch)

    print(f"Epoch {epoch}/{epochs}: D Loss: {D_loss.item()}, G Loss: {adv_loss.item()}, rec loss: {rec_loss.item()}")

    writer.add_scalar("Discriminator Loss", D_loss.item(), epoch)
    writer.add_scalar("Generator Loss", adv_loss.item(), epoch)
    writer.add_scalar("Reconstruction Loss", rec_loss.item(), epoch)

    if epoch % 200 == 0:
        fid_metric.update((real_imgs * 128. + 128.).to(torch.uint8).broadcast_to((256, 3, 28, 28)), real = True)
        fid_metric.update((fake_imgs * 128. + 128.).to(torch.uint8).broadcast_to((256, 3, 28, 28)), real = False)
        fid_score = fid_metric.compute()
        print(f"FID: {fid_score}")
        fid_metric.reset()

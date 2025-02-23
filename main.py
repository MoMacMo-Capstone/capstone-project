import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib
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
    def __init__(self, channels):
        super(DBlock, self).__init__()
        self.block = nn.Sequential(
            ResidualBlock(channels),
            ResidualBlock(channels),
        )

    def forward(self, z):
        return nn.functional.interpolate(self.block(z), scale_factor=0.5, mode="bilinear")

class GBlock(nn.Module):
    def __init__(self, channels, resolution=None):
        super(GBlock, self).__init__()
        self.block = nn.Sequential(
            ResidualBlock(channels),
            ResidualBlock(channels),
        )
        self.resolution = resolution

    def forward(self, z):
        out = self.block(z)

        if self.resolution:
            return nn.functional.interpolate(out, size=self.resolution, mode="bilinear")
        else:
            return nn.functional.interpolate(out, scale_factor=2., mode="bilinear")

class Generator(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(input_dim, latent_dim, 4),
            GBlock(latent_dim),
            GBlock(latent_dim),
            GBlock(latent_dim, (28, 28)),
            nn.Conv2d(latent_dim, 1, 1, bias=False),
        )

    def forward(self, z):
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(self, latent_dim):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, latent_dim, 1),
            DBlock(latent_dim),
            DBlock(latent_dim),
            DBlock(latent_dim),
            nn.Flatten(),
            nn.Linear(latent_dim * 9, 1),
        )

    def forward(self, img):
        return self.model(img)

def zero_centered_gradient_penalty(Samples, Critics):
    Gradient, = torch.autograd.grad(outputs=Critics.sum(), inputs=Samples, create_graph=True)
    return Gradient.square().sum([1, 2, 3])

def generator_loss(discriminator, fake_samples, real_logits):
    fake_logits = discriminator(fake_samples)

    relativistic_logits = fake_logits - real_logits.detach()
    adversarial_loss = nn.functional.softplus(-relativistic_logits)

    return adversarial_loss.mean()

def discriminator_loss(discriminator, fake_samples, real_samples, real_logits, gamma):
    fake_samples = fake_samples.detach().requires_grad_(True)

    fake_logits = discriminator(fake_samples)

    r1_penalty = zero_centered_gradient_penalty(real_samples, real_logits)
    r2_penalty = zero_centered_gradient_penalty(fake_samples, fake_logits)

    relativistic_logits = real_logits - fake_logits
    adversarial_loss = nn.functional.softplus(-relativistic_logits)

    discriminator_loss = adversarial_loss + (gamma / 2) * (r1_penalty + r2_penalty)
    return discriminator_loss.mean()

def image_rows(imgs, num_images=10):
    np = [img.detach().cpu().numpy() for img in imgs]
    _, axes = plt.subplots(len(np), num_images, figsize=(10, 2))
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
input_dim = 1024
batch_size = 256
latent_dim = 16

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

G = Generator(input_dim, latent_dim).to(device)
D = Discriminator(latent_dim).to(device)

optimizer_G = optim.AdamW(G.parameters(), lr=2e-4, betas=(0.0, 0.9))
optimizer_D = optim.AdamW(D.parameters(), lr=2e-4, betas=(0.0, 0.9))
loss_function = nn.BCELoss()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.,), (1.,))
])
train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
fid_metric = FrechetInceptionDistance(feature = 2048)

def interpolate(epoch, start, end):
    a = epoch / (epochs - 1)
    return start * (1 - a) + end * a

for epoch in range(epochs):
    optimizer_D.zero_grad()

    real_imgs = next(iter(train_loader))[0]
    real_imgs = real_imgs.to(device).detach().requires_grad_(True)
    # conditioning = real_imgs.clone()
    # conditioning[:,:,9:21,9:21] = 0

    fake_noise = torch.randn(batch_size, input_dim, 1, 1, device=device)
    fake_imgs = G(fake_noise)

    real_logits = D(real_imgs)

    optimizer_G.zero_grad()
    G_loss = generator_loss(D, fake_imgs, real_logits)
    G_loss.backward()
    optimizer_G.step()

    optimizer_D.zero_grad()
    D_loss = discriminator_loss(D, fake_imgs, real_imgs, real_logits, 0.1)
    D_loss.backward()
    optimizer_D.step()

    if epoch % 25 == 0:
        save_image_rows(f"fig_1_{epoch+1:05}.png", [real_imgs, fake_imgs])

    print(f"Epoch {epoch+1}/{epochs}: D Loss: {D_loss.item()}, G Loss: {G_loss.item()}")

    if epoch % 200 == 0:
        fid_metric.update((real_imgs * 128. + 128.).to(torch.uint8).broadcast_to((256, 3, 28, 28)), real = True)
        fid_metric.update((fake_imgs * 128. + 128.).to(torch.uint8).broadcast_to((256, 3, 28, 28)), real = False)
        fid_score = fid_metric.compute()
        print(f"FID: {fid_score}")
        fid_metric.reset()


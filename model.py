import torch
import torch.nn as nn

resolution = (64, 64)
latent_dim = 64
mean_stdev_latent_dim = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def noise_to_inpainting(noise, mean, stdev, mask):
    inpainting = noise * stdev + mean
    return torch.where(mask, inpainting, mean)

def inpainting_to_noise(image, mean, stdev, mask):
    noise = ((image - mean) / stdev).clamp(-3, 3)
    return noise.masked_fill(~mask, 0)

def abs_norm(images):
    return images.abs().amax(2, keepdim=True).amax(3, keepdim=True) + 1e-9

def abs_normalize(images):
    return images / abs_norm(images)

def color_images(images):
    images = abs_normalize(images) * 2
    return torch.cat([images, images.abs() - 1, -images], dim=1).clamp(0, 1)

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

class MeanEstimator(nn.Module):
    def __init__(self):
        super(MeanEstimator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(2, mean_stdev_latent_dim, 1, bias=False),
            UNET(mean_stdev_latent_dim, resolution),
            nn.Conv2d(mean_stdev_latent_dim, 1, 1, bias=False),
        )

    def forward(self, original, mask):
        original = original.masked_fill(mask, 0)
        norm = abs_norm(original)

        inpainted = self.model(torch.cat([original / norm, mask], dim = 1))
        inpainted = inpainted * norm

        return torch.where(mask, inpainted, original)

class VarEstimator(nn.Module):
    def __init__(self):
        super(VarEstimator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(2, mean_stdev_latent_dim, 1, bias=False),
            UNET(mean_stdev_latent_dim, resolution),
            nn.Conv2d(mean_stdev_latent_dim, 1, 1, bias=False),
        )

    def forward(self, mean, mask):
        norm = abs_norm(mean)

        output = self.model(torch.cat([mean / norm, mask], dim = 1))
        output = output.abs() + 1e-9
        output = output * norm

        return output.masked_fill(~mask, 0)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, latent_dim, 1, bias=False),
            UNET(latent_dim, resolution, inject_noise=True),
            nn.Conv2d(latent_dim, 1, 1, bias=False),
        )

    def forward(self, mean, stdev, mask):
        norm = abs_norm(mean)

        noise = self.model(torch.cat([mean / norm, stdev / norm, mask], dim = 1))

        return noise.masked_fill(~mask, 0)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(4, latent_dim, 1, bias=False),
            UNET(latent_dim, resolution),
        )
        self.linear = nn.Linear(latent_dim, 1)

    def forward(self, noise, mean, stdev, mask):
        z = torch.cat([noise, mean.detach(), stdev.detach(), mask], dim=1)
        z = self.convs(z).mean((2, 3))
        return self.linear(z)

class CombinedGenerator(nn.Module):
    def __init__(self, filename=None):
        super(CombinedGenerator, self).__init__()
        self.mean_estimator = MeanEstimator()
        self.stdev_estimator = VarEstimator()
        self.generator = Generator()

        if filename:
            checkpoint = torch.load(filename)
            self.mean_estimator.load_state_dict(checkpoint["mean"])
            self.stdev_estimator.load_state_dict(checkpoint["stdev"])

            if "generator" in checkpoint:
                self.generator.load_state_dict(checkpoint["generator"])

    def forward(self, original, mask):
        mean = self.mean_estimator(original, mask).detach()
        stdev = self.stdev_estimator(mean, mask).detach()
        noise = self.generator(mean, stdev, mask)
        return noise_to_inpainting(noise, mean, stdev, mask)

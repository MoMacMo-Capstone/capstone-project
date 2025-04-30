import torch
from torchmetrics.image.fid import FrechetInceptionDistance

import read_seismic_data

def color_images(images):
    images = images / (images.abs().amax(2, keepdim=True).amax(3, keepdim=True) + 1e-9)
    images = images * 2
    return torch.cat([images, images.abs() - 1, -images], dim=1).clamp(0, 1)

def prepare_for_fid(imgs):
    return (color_images(imgs) * 256).clamp(0, 255).to(torch.uint8)

resolution = (128, 128)
batch_size = 256

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# transform = transforms.Compose([
    # transforms.ToTensor(),
    # transforms.Normalize((0.5,), (0.5,))
# ])
# train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
# train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
fid_metric_acc = FrechetInceptionDistance(feature = 2048).to(device)
fid_metric = FrechetInceptionDistance(feature = 2048).to(device)

exit()

epoch = 0

while True:
    epoch += 1

    imgs_1 = read_seismic_data.get_chunks(batch_size, resolution[0])
    imgs_1 = torch.tensor(imgs_1, device=device)

    imgs_2 = read_seismic_data.get_chunks(batch_size, resolution[0])
    imgs_2 = torch.tensor(imgs_2, device=device)

    print(imgs_1.shape, prepare_for_fid(imgs_1).shape)
    fid_metric_acc.update(prepare_for_fid(imgs_1), real = True)
    fid_metric.reset()
    fid_metric.merge_state(fid_metric_acc)
    fid_metric.update(prepare_for_fid(imgs_2), real = False)
    fid_score = fid_metric.compute()
    print(f"FID: {fid_score}")

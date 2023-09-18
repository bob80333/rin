# build 2 folders, one of dataset, one of generated images

import os

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from tqdm import tqdm, trange

from generate import generate_n_k

# build dataset folder

N_IMAGES = 50_000

os.makedirs('dataset_images', exist_ok=True)

dataset = CIFAR10(root='data', download=True,
                  transform=transforms.ToTensor(), train=True)

dl = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

for image in dl:
    image = image[0]
    image = image * 255
    image = image.permute(1, 2, 0)
    image = image.to(torch.uint8)
    torchvision.utils.save_image(image, f'dataset_images/{i}.png')


# build generated images folder

os.makedirs('generated_images', exist_ok=True)
# 10 classes, 0-9, 5k of each

generate_n_k('model.pt', 5000, 32, 0, 'generated_images')
generate_n_k('model.pt', 5000, 32, 1, 'generated_images')
generate_n_k('model.pt', 5000, 32, 2, 'generated_images')
generate_n_k('model.pt', 5000, 32, 3, 'generated_images')
generate_n_k('model.pt', 5000, 32, 4, 'generated_images')
generate_n_k('model.pt', 5000, 32, 5, 'generated_images')
generate_n_k('model.pt', 5000, 32, 6, 'generated_images')
generate_n_k('model.pt', 5000, 32, 7, 'generated_images')
generate_n_k('model.pt', 5000, 32, 8, 'generated_images')
generate_n_k('model.pt', 5000, 32, 9, 'generated_images')

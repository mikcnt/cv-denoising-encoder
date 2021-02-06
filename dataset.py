import os
import random
from PIL import Image

import cv2
import numpy as np
from torch.utils.data import Dataset

import utils.noise as noise
from utils.data import find_paths, t_path

random.seed(42069)

# A class defining the dataset
class ImageDataset(Dataset):
    def __init__(
        self,
        images_folder,
        g_min=0.08,
        g_max=0.15,
        p_min=0.1,
        p_max=0.2,
        s_min=0.1,
        s_max=0.2,
        transform=None,
    ):
        super().__init__()
        files = os.listdir(images_folder)
        self.image_paths = [
            images_folder + "/" + file
            for file in files
            if file.endswith((".jpg", ".png"))
        ]
        self.g_min = g_min
        self.g_max = g_max
        self.p_min = p_min
        self.p_max = p_max
        self.s_min = s_min
        self.s_max = s_max
        self.transform = transform

    # Returns the number of samples, it is used for iteration porpuses
    def __len__(self):
        return len(self.image_paths)

    # Returns a random sample for training(generally)
    def __getitem__(self, idx):
        # Load RANDOM clean image into memory...
        image_path = self.image_paths[idx]
        clean_image = np.array(Image.open(image_path)) / 255
        noisy_image = clean_image.copy()
        
        noisy_image = cv2.resize(noisy_image, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
        clean_image = cv2.resize(clean_image, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
        
        noisy_image = noise.pepper(
            noisy_image, threshold=1, amount=random.uniform(self.p_min, self.p_max)
        )
        noisy_image = noise.gaussian(
            noisy_image, amount=random.uniform(self.g_min, self.g_max)
        )
        noisy_image = noise.salt(
            noisy_image, amount=random.uniform(self.s_min, self.s_max)
        )

        clean_image = clean_image.astype(np.float32)
        noisy_image = noisy_image.astype(np.float32)

        if self.transform:
            noisy_image = self.transform(noisy_image)
            clean_image = self.transform(clean_image)

        return noisy_image, clean_image


class RenderDataset(Dataset):
    def __init__(
        self,
        images_folder,
        transform=None,
    ):
        super().__init__()
        self.images_folder = images_folder
        self.image_paths = sorted(find_paths(images_folder))
        self.transform = transform

    # Returns the number of samples, it is used for iteration porpuses
    def __len__(self):
        return len(self.image_paths)

    # Returns a random sample for training(generally)
    def __getitem__(self, idx):
        # Load RANDOM clean image into memory...
        image_path = self.image_paths[idx]
        noisy_image = np.array(Image.open(image_path))
        clean_image = np.array(Image.open(t_path(self.images_folder, image_path)))

        if self.transform:
            augmentations = self.transform(image=noisy_image, image0=clean_image)

            noisy_image = augmentations["image"]
            clean_image = augmentations["image0"]

        clean_image = (clean_image / 255).astype(np.float32)
        noisy_image = (noisy_image / 255).astype(np.float32)

        return noisy_image, clean_image
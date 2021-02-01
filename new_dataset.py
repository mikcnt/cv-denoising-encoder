import os
import random

import cv2
import numpy as np
from torch.utils.data import Dataset

import utils.noise as noise
from utils.data import find_paths, t_path

from PIL import Image


random.seed(42069)


# A class defining the dataset
class ImageDataset(Dataset):
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
        noisy_image = np.array(Image.open(image_path)) / 255
        clean_image = np.array(Image.open(t_path(self.images_folder, image_path))) / 255

        clean_image = clean_image.astype(np.float32)
        noisy_image = noisy_image.astype(np.float32)

        if self.transform:
            noisy_image = self.transform(noisy_image)
            clean_image = self.transform(clean_image)

        return noisy_image, clean_image

import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class ColoredMangaDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        super(ColoredMangaDataset, self).__init__()
        self.image_dir = image_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        image = np.array(Image.open(img_path).convert("RGB"))
        augmentation = self.transform(image=image)
        image = augmentation["image"]
        gray_image = image.clone().detach()
        h, w = gray_image.shape[1], gray_image.shape[2]
        gray_image = gray_image.reshape(3, -1)
        gray_image = (
            gray_image[0] * 0.299 + gray_image[1] * 0.587 + gray_image[2] * 0.114
        )
        gray_image = gray_image.reshape(1, h, w)
        gray_image = gray_image.clone().detach()
        return gray_image, image

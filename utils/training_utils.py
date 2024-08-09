import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def list_directories(folder_path):
    return [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]


class CustomImageDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []
        self.classes = sorted(list_directories(root_dir))

        for idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            for image_name in os.listdir(class_dir):
                self.image_paths.append(os.path.join(class_dir, image_name))
                self.labels.append(idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        image = torch.tensor(image)
        label = self.labels[idx]

        return image, label
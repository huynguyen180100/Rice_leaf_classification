import os
from numpy import imag
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
import pytorch_lightning as pl


class MyImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        return super(MyImageFolder, self).__getitem__(index), self.imgs[index][0]

class FlowerDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./data",
                        image_size = (32,32),
                        batch_size: int=32,
                        num_workers: int=2):
        super(FlowerDataModule, self).__init__()

        self.data_dir = data_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)

    @property
    def normalize(self):
        return transforms.Normalize(mean=self.mean.tolist(), std=self.std.tolist())

    @property
    def train_transforms(self):
        return transforms.Compose([
            transforms.Resize(self.image_size, interpolation=InterpolationMode.BICUBIC),
            transforms.RandomRotation(15),
            transforms.RandomAffine(translate=(0.08, 0.1), degrees=15),
            transforms.ColorJitter(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomInvert(p=0.5),
            transforms.ToTensor(),
            self.normalize
        ])

    @property
    def val_transforms(self):
        return transforms.Compose([
            transforms.Resize(self.image_size, interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            self.normalize
        ])

    def _dataloader(self, mode):
        is_shuffle = False
        if mode == "train":
            is_shuffle = True
            train_path = os.path.join(self.data_dir, "train")
            data = MyImageFolder(root=train_path, transform=self.train_transforms)
        if mode == "val":
            valid_path = os.path.join(self.data_dir, "val")
            data = MyImageFolder(root=valid_path, transform=self.val_transforms)
        return DataLoader(dataset=data,
                        batch_size=self.batch_size,
                        num_workers=self.num_workers,
                        shuffle=is_shuffle)

    def train_dataloader(self):
        return self._dataloader(mode="train")

    def val_dataloader(self):
        return self._dataloader(mode="val")
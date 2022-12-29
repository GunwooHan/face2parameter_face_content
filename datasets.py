import torch
from PIL import Image


class P2FFacialContentDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, image_transform=None, label_transform=None):
        self.images = images
        self.labels = labels
        self.image_transform = image_transform
        self.label_transform = label_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        label = Image.open(self.labels[idx])

        if self.image_transform:
            image = self.image_transform(image)

        if self.label_transform:
            label = self.label_transform(label).squeeze(0).long()

        return image, label

import os

import kornia
import cv2
import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torchmetrics.functional import jaccard_index

from torchvision.models.resnet import resnet50


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        self.skip = True if in_channels != out_channels else False
        self.downsample = downsample

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=3, padding=1, stride=2 if self.downsample else 1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        if self.skip or self.downsample:
            self.identity = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2 if self.downsample else 1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, tensor):
        x = self.conv(tensor)

        if self.skip or self.downsample:
            tensor = self.identity(tensor)
        return x + tensor


class ResNet50Seg(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.block1 = nn.Sequential(
            ResBlock(64, 64, downsample=True),
            ResBlock(64, 64),
            ResBlock(64, 128),
        )
        self.block2 = nn.Sequential(
            ResBlock(128, 128),
            ResBlock(128, 128),
            ResBlock(128, 128),
            ResBlock(128, 256),
        )
        self.block3 = nn.Sequential(
            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 512),
        )
        self.block4 = nn.Sequential(
            ResBlock(512, 512),
            ResBlock(512, 512),
            ResBlock(512, 512),
        )
        self.last_conv = nn.Conv2d(512, num_classes, kernel_size=1)

    def forward(self, tensor):
        x = self.conv(tensor)
        x = self.maxpool(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.last_conv(x)
        return x


class FacialContentSegmentation(pl.LightningModule):
    def __init__(self, args):
        super(FacialContentSegmentation, self).__init__()
        self.args = args
        self.loss_fn = nn.CrossEntropyLoss()
        self.model = ResNet50Seg(self.args.num_classes)
        self.label_name = {0: "backgroud", 1: "skin", 2: "nose", 3: "eye", 4: "eyebrow", 5: "upper_lip", 6: "lower_lip",
                           7: "hair"}

    def forward(self, parameter):
        return self.model(parameter)

    def configure_optimizers(self):
        opt_g = optim.SGD(self.model.parameters(), lr=self.args.learning_rate, momentum=0.9)

        scheduler_g = lr_scheduler.StepLR(opt_g, step_size=50, gamma=0.9, verbose=True)

        return {"optimizer": opt_g, "lr_scheduler": {"scheduler": scheduler_g, "interval": "epoch"}}

    def training_step(self, train_batch, batch_idx):
        images, masks = train_batch

        pred = self.model(images)
        loss = self.loss_fn(pred, masks)
        jaccard_index_value = jaccard_index(pred.argmax(dim=1), masks, task="multiclass",
                                            num_classes=self.args.num_classes)

        self.log('train/loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train/jaccard_index_value', jaccard_index_value, on_epoch=True, prog_bar=True, sync_dist=True)
        return {"loss": loss}

    def validation_step(self, val_batch, batch_idx):
        images, masks = val_batch

        pred = self.model(images)
        loss = self.loss_fn(pred, masks)
        jaccard_index_value = jaccard_index(pred.argmax(dim=1), masks, task="multiclass",
                                            num_classes=self.args.num_classes)

        self.log('val/loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val/jaccard_index_value', jaccard_index_value, on_epoch=True, prog_bar=True, sync_dist=True)

        if batch_idx == 0:
            sample_count = 4 if images.size(0) > 4 else images.size(0)
            sample_images = F.interpolate(images[:sample_count], (64, 64), mode="nearest")
            sample_masks = masks[:sample_count].unsqueeze(1).type(torch.uint8)
            sample_preds = pred[:sample_count].argmax(dim=1).unsqueeze(1).type(torch.uint8)

            self.logger.log_image(key='sample_images', images=[sample_images[i] for i in range(sample_count)],
                                  caption=[self.current_epoch + 1 for _ in range(sample_count)],
                                  masks=[{
                                      "prediction": {
                                          "mask_data": sample_preds[i].squeeze(0).cpu().numpy(),
                                          "class_labels": self.label_name
                                      },
                                      "ground_truth": {
                                          "mask_data": sample_masks[i].squeeze(0).cpu().numpy(),
                                          "class_labels": self.label_name
                                      },
                                  } for i in range(sample_count)])
        return {"loss": loss}


if __name__ == '__main__':
    model = ResNet50Seg(10)
    inputs = torch.randn(2, 3, 512, 512)
    outputs = model(inputs)
    print(outputs.shape)
    # torch.Size([2, 64, 256, 256])
    # torch.Size([2, 256, 128, 128])
    # torch.Size([2, 512, 64, 64])
    # torch.Size([2, 1024, 32, 32])
    # torch.Size([2, 2048, 16, 16])

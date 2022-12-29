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


class ResNet50Seg(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.encoder = timm.create_model('resnet50', features_only=True)
        # self.head = nn.Sequential(
        #     nn.BatchNorm2d(1024),
        #     nn.ReLU(),
        #     nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(),
        #     nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #     nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #     nn.Conv2d(128, num_classes, kernel_size=7, stride=1, padding=3),
        # )
        self.dec1 = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Upsample(scale_factor=2)
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Upsample(scale_factor=2)
        )
        self.dec3 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.last_conv = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, kernel_size=7, stride=1, padding=3),
        )

    def forward(self, tensor):
        features = self.encoder(tensor)
        x = self.dec1(features[4])
        x = torch.cat([x, features[3]], dim=1)
        x = self.dec2(x)
        x = torch.cat([x, features[2]], dim=1)
        x = self.dec3(x)
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
    for o in outputs:
        print(o.shape)
    # torch.Size([2, 64, 256, 256])
    # torch.Size([2, 256, 128, 128])
    # torch.Size([2, 512, 64, 64])
    # torch.Size([2, 1024, 32, 32])
    # torch.Size([2, 2048, 16, 16])

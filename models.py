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

from torchvision.models import resnet50, ResNet50_Weights


def _make_model_resnet50seg(num_classes, pretraiend=False):
    base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretraiend else None)

    base_model.layer1[0].conv2.stride = (2, 2)
    base_model.layer1[0].downsample[0].stride = (2, 2)
    base_model.layer2[0].conv2.stride = (1, 1)
    base_model.layer2[0].downsample[0].stride = (1, 1)
    base_model.layer3[0].conv2.stride = (1, 1)
    base_model.layer3[0].downsample[0].stride = (1, 1)
    base_model.layer4[0].conv2.stride = (1, 1)
    base_model.layer4[0].downsample[0].stride = (1, 1)

    model = nn.Sequential(
        base_model.conv1,
        base_model.bn1,
        base_model.relu,
        base_model.maxpool,
        base_model.layer1,
        base_model.layer2,
        base_model.layer3,
        base_model.layer4,
        nn.Conv2d(2048, num_classes, kernel_size=1)
    )
    return model

class FacialContentSegmentation(pl.LightningModule):
    def __init__(self, args):
        super(FacialContentSegmentation, self).__init__()
        self.args = args
        self.loss_fn = nn.CrossEntropyLoss()
        self.model = _make_model_resnet50seg(self.args.num_classes, self.args.model_pretrained)
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
    model = _make_model_resnet50seg(10, True)
    inputs = torch.randn(2, 3, 512, 512)
    outputs = model(inputs)
    print(outputs.shape)
    model = _make_model_resnet50seg(10, False)
    inputs = torch.randn(2, 3, 512, 512)
    outputs = model(inputs)
    print(outputs.shape)

    # torch.Size([2, 64, 256, 256])
    # torch.Size([2, 256, 128, 128])
    # torch.Size([2, 512, 64, 64])
    # torch.Size([2, 1024, 32, 32])
    # torch.Size([2, 2048, 16, 16])

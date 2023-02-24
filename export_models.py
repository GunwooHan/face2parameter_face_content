import argparse
import glob
import os

import numpy as np
import pytorch_lightning as pl
import torchvision.transforms
import wandb
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split

from datasets import P2FFacialContentDataset
from models import FacialContentSegmentation

parser = argparse.ArgumentParser()

# 데이터 관련 설정
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--data_dir', type=str, default='dataset')

# 모델 관련 설정
parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--precision', type=int, default=32)
parser.add_argument('--num_workers', type=int, default=16)
parser.add_argument('--project', type=str, default='f2p_facial_content')
parser.add_argument('--name', type=str, default='resnet50')

# 학습 관련 설정

parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--num_classes', type=int, default=8)
parser.add_argument('--model_pretrained', type=bool, default=True)

args = parser.parse_args()

if __name__ == '__main__':
    model = FacialContentSegmentation(args).load_from_checkpoint(
        "checkpoints/resnet50_torchvision_pretrained_val/jaccard_index_value=0.2625.ckpt", args=args)
    torch.save(model.model, "resnet50_facial_content.pth")

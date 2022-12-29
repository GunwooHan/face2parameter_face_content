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
args = parser.parse_args()

if __name__ == '__main__':
    pl.seed_everything(args.seed)
    images = np.array(sorted(glob.glob(os.path.join(args.data_dir, 'data', '*'))))
    masks = np.array(sorted(glob.glob(os.path.join(args.data_dir, 'label', '*'))))

    train_images, val_images, train_masks, val_masks = train_test_split(images, masks, test_size=0.2)

    wandb_logger = WandbLogger(project=args.project, name=args.name)
    checkpoint_callback = ModelCheckpoint(
        monitor="val/jaccard_index_value",
        dirpath="checkpoints",
        filename=f"{args.name}_" + "{val/jaccard_index_value:.4f}",
        save_top_k=3,
        mode="max",
    )

    # early_stop_callback = EarlyStopping(monitor="val/loss", min_delta=0.00, patience=50, verbose=True, mode="min")

    image_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((512, 512), interpolation=torchvision.transforms.InterpolationMode.NEAREST,),
            torchvision.transforms.ToTensor()
        ]
    )
    label_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((64, 64), interpolation=torchvision.transforms.InterpolationMode.NEAREST,),
            torchvision.transforms.PILToTensor(),
        ]
    )

    model = FacialContentSegmentation(args)

    train_ds = P2FFacialContentDataset(train_images, train_masks, image_transform, label_transform)
    train_dataloader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size,
                                                   num_workers=args.num_workers, shuffle=True,
                                                   drop_last=True)

    val_ds = P2FFacialContentDataset(val_images, val_masks, image_transform, label_transform)
    val_dataloader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size,
                                                 num_workers=args.num_workers)

    trainer = pl.Trainer(accelerator='gpu',
                         devices=args.gpus,
                         precision=args.precision,
                         max_epochs=args.epochs,
                         # log_every_n_steps=1,
                         strategy='ddp',
                         # num_sanity_val_steps=0,
                         # limit_train_batches=5,
                         # limit_val_batches=1,
                         logger=wandb_logger,
                         callbacks=[checkpoint_callback])

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    wandb.finish()

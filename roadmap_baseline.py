import os
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
import torchvision.models as models

from data_helper import UnlabeledDataset, LabeledDataset
from helper import collate_fn, draw_box

import pytorch_lightning as pl


class RoadMap(pl.LightningModule):

    def __init__(self):
        super().__init__()

        output_dim = 800*800

        self.feature_extractor = models.resnet50(
                                    pretrained=True,
                                    num_classes=1000)
        self.feature_extractor.fc = Identity()
        self.feature_extractor.eval()

        # use the pretrained model to predict binary roadmap
        self.linear_1 = nn.Linear(2048, output_dim)

    def forward(self, x):
        # called with self(x)
        representations = self.feature_extractor(x)
        outputs = F.sigmoid(self.linear_1(representations))
        return outputs

    def training_step(self, batch, batch_idx):
        # REQUIRED
        sample, target, road_image = batch

        # change samples from tuple with length batch size containing 6x3xHxW to batch_sizex6x3xHxW
        x = torch.stack(sample, dim=0)

        # reorder 6 images for each sample
        x = x[:, [0, 1, 2, 5, 4, 3]]

        # reshape to wide format - stitch 6 images side by side
        x = x.permute(0, 2, 1, 3, 4).reshape(batch_size, 3, 256, -1)

        # get the road-image y with shape batch_sizex800x800
        y = torch.stack(road_image, dim=0)

        # forward pass to calculate outputs
        outputs = self(x)

        # flatten y and outputs for binary cross entropy
        outputs = outputs.view(outputs.size(0), -1)
        y = y.view(y.size(0), -1).float()

        loss = F.binary_cross_entropy(outputs, y)

        tensorboard_logs = {'train_loss': loss}

        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        y = torch.round(torch.rand(batch_size, 800*800))
        y_hat = self.forward(x)
        return {'val_loss': F.binary_cross_entropy(y_hat, y)}

    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)
        return {'test_loss': F.cross_entropy(y_hat, y)}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss}
        return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def prepare_data(self):
        self.cifar_train = CIFAR10(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
        self.cifar_test = CIFAR10(os.getcwd(), train=False, download=True, transform=transforms.ToTensor())

    def train_dataloader(self):
        image_folder = 'data'
        annotation_csv = 'data/annotation.csv'
        transform = transforms.ToTensor()
        labeled_trainset = LabeledDataset(image_folder=image_folder,
                                          annotation_file=annotation_csv,
                                          scene_index=labeled_scene_index,
                                          transform=transform,
                                          extra_info=False
                                          )
        loader = DataLoader(labeled_trainset, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn)
        #loader = DataLoader(self.mnist_train, batch_size=32)
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.cifar_test, batch_size=batch_size)
        return loader

    def test_dataloader(self):
        loader = DataLoader(self.cifar_test, batch_size=batch_size)
        return loader


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


if __name__ == '__main__':
    #parser = ArgumentParser()
    #parser = pl.Trainer.add_argparse_args(parser)
    #parser = VAE.add_model_specific_args(parser)
    #args = parser.parse_args()

    unlabeled_scene_index = np.arange(106)
    labeled_scene_index = np.arange(106, 134)
    batch_size = 2

    model = RoadMap()
    trainer = pl.Trainer(num_sanity_val_steps=1)
    trainer.fit(model)
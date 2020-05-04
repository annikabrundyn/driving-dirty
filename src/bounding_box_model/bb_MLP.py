'''
This script should contain the model that takes the images, converts the bounding box output to images,
does feature extraction, tries to predict bounding box image and then converts the image back to coordinate
predictions.
'''

import os
import random
import numpy as np
import torch

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from argparse import ArgumentParser, Namespace

from pytorch_lightning import LightningModule, Trainer
from test_tube import HyperOptArgumentParser

from src.utils.helper import collate_fn, plot_image, log_rm_images
from src.utils.data_helper import LabeledDataset

from src.autoencoder.autoencoder import BasicAE

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.utils.helper import draw_box


class Boxes(LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.ae = BasicAE.load_from_checkpoint(self.hparams.pretrained_path)
        self.frozen = True
        self.ae.freeze()
        self.ae.decoder = None

        # calculate output dim = padded no of bbs * 8 coordinates
        self.output_dim = self.hparams.max_bb * 8

        # MLP layers
        self.fc1 = nn.Linear(self.ae.latent_dim, self.output_dim)

    def wide_stitch_six_images(self, sample):
        # change from tuple len([6 x 3 x H x W]) = b --> tensor [b x 6 x 3 x H x W]
        x = torch.stack(sample, dim=0)

        # reorder order of 6 images (in first dimension) to become 180 degree view
        x = x[:, [0, 1, 2, 5, 4, 3]]

        # rearrange axes and reshape to wide format
        b, num_imgs, c, h, w = x.size()
        x = x.permute(0, 2, 3, 1, 4).reshape(b, c, h, -1)
        return x

    def pad_bb_coordinates(self, target):
        # loop over the items in the batch
        # target is a tuple of len batch_size
        output = torch.zeros(self.hparams.batch_size, self.output_dim)
        for i, sample in enumerate(target):
            flat_coords = sample['bounding_box'].reshape(-1)
            output[i, 0:flat_coords.size(0)] = flat_coords
        return output

    def forward(self, x):
        # called with self(x)
        x = self.wide_stitch_six_images(x)

        representations = self.ae.encoder(x)

        # now run through MLP
        y = self.fc1(representations)
        return y

    def _run_step(self, batch, batch_idx, step_name):
        sample, target, road_image = batch

        # transform the target from bb coordinates into padded tensors
        target_bb = self.pad_bb_coordinates(target)

        # forward pass to find predicted bb tensor
        pred_bb = self(sample)

        # calculate the MSE loss between coordinates
        loss = F.mse_loss(target_bb, pred_bb)

        # draw coordinates to visualize
        # every however many epochs we look at inputs + predictions
        if batch_idx % self.hparams.output_img_freq == 0:
            x = self.wide_stitch_six_images(sample)

            # reshape target
            # (b, 8*100) -> (8*100)
            target_bb_eg = target_bb[0]
            pred_bb_eg = pred_bb[0]

            # (8*100) -> (100, 2, 4)
            # TODO: check
            target_bb_eg = target_bb_eg.reshape(self.hparams.max_bb, 2, 4)
            pred_bb_eg = pred_bb_eg.reshape(self.hparams.max_bb, 2, 4)

            # (100, 2, 4) -> (b=1, 1, 755, 756)
            y_hat_boxes = plot_image(pred_bb_eg)
            y_boxes = plot_image(target_bb_eg)

            log_rm_images(self, x, y_boxes, y_hat_boxes, step_name)

        return loss, target_bb, pred_bb


    def training_step(self, batch, batch_idx):

        #if self.current_epoch >= 30 and self.frozen:
        #    self.frozen=False
        #    self.ae.unfreeze()

        train_loss, _, _ = self._run_step(batch, batch_idx, step_name='train')
        train_tensorboard_logs = {'train_loss': train_loss}
        return {'loss': train_loss, 'log': train_tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        val_loss, target_rm, pred_rm = self._run_step(batch, batch_idx, step_name='valid')

        return {'val_loss': val_loss}

    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_tensorboard_logs = {'avg_val_loss': avg_val_loss}
        return {'val_loss': avg_val_loss, 'log': val_tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def prepare_data(self):
        image_folder = self.hparams.link
        annotation_csv = self.hparams.link + '/annotation.csv'

        transform = torchvision.transforms.ToTensor()

        labeled_dataset = LabeledDataset(image_folder=image_folder,
                                         annotation_file=annotation_csv,
                                         scene_index=np.arange(106, 134),
                                         transform=transform,
                                         extra_info=False)

        trainset_size = round(0.8 * len(labeled_dataset))
        validset_size = round(0.2 * len(labeled_dataset))

        # split train + valid at the sample level (ie 6 image collections) not scene/video level
        self.trainset, self.validset = torch.utils.data.random_split(labeled_dataset,
                                                                     lengths = [trainset_size, validset_size])

    def train_dataloader(self):
        loader = DataLoader(self.trainset,
                            batch_size=self.hparams.batch_size,
                            shuffle=True,
                            num_workers=2,
                            collate_fn=collate_fn)
        return loader

    def val_dataloader(self):
        # don't shuffle validation batches
        loader = DataLoader(self.validset,
                            batch_size=self.hparams.batch_size,
                            shuffle=False,
                            num_workers=2,
                            collate_fn=collate_fn)
        return loader

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = HyperOptArgumentParser(parents=[parent_parser], add_help=False)

        # want to optimize this parameter
        #parser.opt_list('--batch_size', type=int, default=16, options=[16, 10, 8], tunable=False)
        parser.opt_list('--learning_rate', type=float, default=0.005, options=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5], tunable=True)
        parser.add_argument('--batch_size', type=int, default=16)
        parser.add_argument('--max_bb', type=int, default=100)

        # fixed arguments
        parser.add_argument('--link', type=str, default='/Users/annika/Developer/driving-dirty/data')
        parser.add_argument('--pretrained_path', type=str, default='/Users/annika/Developer/driving-dirty/lightning_logs/version_3/checkpoints/epoch=4.ckpt')
        parser.add_argument('--output_img_freq', type=int, default=1000)
        return parser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = Boxes.add_model_specific_args(parser)
    args = parser.parse_args()

    model = Boxes(args)
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model)


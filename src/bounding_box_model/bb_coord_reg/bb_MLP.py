'''
This script should contain the model that takes the images, converts the bounding box output to images,
does feature extraction, tries to predict bounding box image and then converts the image back to coordinate
predictions.
'''

import os
import random
import numpy as np
import torch

random.seed(20200505)
np.random.seed(20200505)
torch.manual_seed(20200505)

import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from argparse import ArgumentParser, Namespace

from pytorch_lightning import LightningModule, Trainer
from test_tube import HyperOptArgumentParser

from src.utils.helper import collate_fn, plot_image, log_bb_images, plot_all_boxes_new
from src.utils.data_helper import LabeledDataset

from src.autoencoder.autoencoder import BasicAE

import matplotlib
matplotlib.use('Agg')

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
        self.fc1 = nn.Linear(self.ae.latent_dim, self.output_dim//2)
        self.fc2 = nn.Linear(self.output_dim//2, self.output_dim)

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
        # target is a tuple of len batch_size

        # initialize padded output vector with dim [b, max_bb, 2, 4]
        output = torch.zeros(self.hparams.batch_size, self.hparams.max_bb, 2, 4)

        # loop over the items in the batch
        for i, sample in enumerate(target):
            # dim is [num_bb, 2, 4]
            flat_coords = sample['bounding_box']

            # get the num of bounding boxes in this image
            num_bb = flat_coords.size(0)

            # replace non-zero box coordinates in padded vector
            output[i, 0:num_bb, ...] = flat_coords

        # output has dim [b, max_bb, 2, 4]
        return output

    def forward(self, x):
        # called with self(x)
        x = self.wide_stitch_six_images(x)

        representations = self.ae.encoder(x)

        # now run through MLP
        y = F.relu(self.fc1(representations))
        y = self.fc2(y)

        # reshape the predictions to be:
        # [b, max_bb*2*4] -> [b, max_bb, 2, 4]
        y = y.reshape(y.size(0), self.hparams.max_bb, 2, 4)

        return y

    def _run_step(self, batch, batch_idx, step_name):
        sample, target, road_image = batch

        # transform the target from bb coordinates into padded tensors
        # (b) tuple of dicts -> [b, max_bb, 2, 4]
        target_bb = self.pad_bb_coordinates(target)
        target_bb = target_bb.type_as(sample[0])

        # forward pass to find predicted bb tensor
        # -> [b, max_bb, 4, 2]
        pred_bb = self(sample)

        # every few epochs we visualize inputs + predictions
        if batch_idx % self.hparams.output_img_freq == 0:
            # x dim: [b, 3, 256, 1836]
            x = self.wide_stitch_six_images(sample)

            # take the first image in batch
            # [b, max_bb, 2, 4] -> [max_bb, 2, 4]
            target_bb0 = target_bb[0]
            pred_bb0 = pred_bb[0]

            # have to reshape pred bb from [max_bb*2*4] -> [max_bb, 2, 4]
            # target_bb0 = target_bb0.reshape(self.hparams.max_bb, 2, 4)
            # pred_bb0 = pred_bb0.reshape(self.hparams.max_bb, 2, 4)

            # [100, 2, 4] -> matplotlib figure
            pred_img = plot_all_boxes_new(pred_bb0)
            target_img = plot_all_boxes_new(target_bb0)

            log_bb_images(self, x, target_img, pred_img, step_name)

        # calculate the MSE loss between coordinates
        # note: F.mse_loss calculates element wise MSE loss so I don't have to flatten tensors
        loss = F.mse_loss(target_bb, pred_bb)
        return loss, target_bb, pred_bb

    def training_step(self, batch, batch_idx):
        if self.current_epoch >= self.hparams.unfreeze_epoch_no and self.frozen:
            self.frozen=False
            self.ae.unfreeze()
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
        labeled_scene_index = np.arange(106, 134)
        trainset_size = round(0.8 * len(labeled_scene_index))

        # split into train / validation sets at the scene index level
        # before I did this at the sample level --> this will cause leakage (!!)
        np.random.shuffle(labeled_scene_index)
        train_set_index = labeled_scene_index[:trainset_size]
        valid_set_index = labeled_scene_index[trainset_size:]

        transform = torchvision.transforms.ToTensor()

        # training set
        self.labeled_trainset = LabeledDataset(image_folder=image_folder,
                                               annotation_file=annotation_csv,
                                               scene_index=train_set_index,
                                               transform=transform,
                                               extra_info=False)

        # validation set
        self.labeled_validset = LabeledDataset(image_folder=image_folder,
                                               annotation_file=annotation_csv,
                                               scene_index=valid_set_index,
                                               transform=transform,
                                               extra_info=False)

    def train_dataloader(self):
        loader = DataLoader(self.labeled_trainset,
                            batch_size=self.hparams.batch_size,
                            shuffle=True,
                            num_workers=4,
                            collate_fn=collate_fn)
        return loader

    def val_dataloader(self):
        # don't shuffle validation batches
        loader = DataLoader(self.labeled_validset,
                            batch_size=self.hparams.batch_size,
                            shuffle=False,
                            num_workers=4,
                            collate_fn=collate_fn)
        return loader

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = HyperOptArgumentParser(parents=[parent_parser], add_help=False)

        # want to optimize this parameter
        #parser.opt_list('--batch_size', type=int, default=16, options=[16, 10, 8], tunable=False)
        parser.opt_list('--learning_rate', type=float, default=0.001, options=[1e-3, 1e-4, 1e-5], tunable=True)
        parser.add_argument('--batch_size', type=int, default=16)
        parser.add_argument('--max_bb', type=int, default=100)

        # fixed arguments
        parser.add_argument('--link', type=str, default='/scratch/ab8690/DLSP20Dataset/data')
        parser.add_argument('--pretrained_path', type=str, default='/scratch/ab8690/logs/dd_pretrain_ae/lightning_logs/version_9234267/checkpoints/epoch=42.ckpt')
        parser.add_argument('--output_img_freq', type=int, default=500)
        parser.add_argument('--unfreeze_epoch_no', type=int, default=30)

        return parser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = Boxes.add_model_specific_args(parser)
    args = parser.parse_args()

    model = Boxes(args)
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model)


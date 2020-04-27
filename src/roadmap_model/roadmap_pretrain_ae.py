import random

from argparse import ArgumentParser

import numpy as np

import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from pytorch_lightning import LightningModule, Trainer
from test_tube import HyperOptArgumentParser

from src.utils.data_helper import LabeledDataset
from src.utils.helper import collate_fn

from src.autoencoder.autoencoder import BasicAE

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


class RoadMap(LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.__check_hparams(hparams)
        self.hparams = hparams
        self.output_dim = 800 * 800
        #self.kernel_size = 4
        path_checkpoint ='/Users/annika/Developer/driving-dirty/lightning_logs/version_3/checkpoints/epoch=4.ckpt'
        patched_ckpt_name = self.patch_checkpoint(path_checkpoint)

        # pretrained feature extractor - using our own trained Encoder
        self.ae = BasicAE.load_from_checkpoint(patched_ckpt_name)
        self.ae.freeze()
        self.ae.decoder = None

        # MLP layers: feature embedding --> predict binary roadmap
        self.fc1 = nn.Linear(self.ae.latent_dim, 200000)
        self.fc2 = nn.Linear(200000, self.output_dim)

    def __check_hparams(self, hparams):
        self.batch_size = hparams.batch_size if hasattr(hparams, 'batch_size') else 2
        self.in_channels = hparams.in_channels if hasattr(hparams, 'in_channels') else 3

    def wide_stitch_six_images(self, sample):
        # change from tuple len([6 x 3 x H x W]) = b --> tensor [b x 6 x 3 x H x W]
        x = torch.stack(sample, dim=0)

        # reorder order of 6 images (in first dimension) to become 180 degree view
        x = x[:, [0, 1, 2, 5, 4, 3]]

        # rearrange axes and reshape to wide format
        b, num_imgs, c, h, w = x.size()
        x = x.permute(0, 2, 3, 1, 4).reshape(b, c, h, -1)
        assert x.size(-1) == 6 * 306

        return x

    def forward(self, x):
        # note: can call forward(x) with self(x)
        # first find representations using the pretrained encoder
        representations = self.ae.encoder(x)

        # now run through MLP
        y = F.relu(self.fc1(representations))
        y = F.sigmoid(self.fc2(y))

        return y

    def _run_step(self, batch, batch_idx, step_name):
        sample, target, road_image = batch

        # wide stitch the 6 images in sample
        x = self.wide_stitch_six_images(sample)

        # change target roadmap from tuple len([800 x 800]) = b --> tensor [b x 800 x 800]
        target_rm = torch.stack(road_image, dim=0)

        # forward pass to find predicted roadmap
        pred_rm = self(x)

        # every 10 epochs we look at inputs + predictions
        if batch_idx % 10 == 0:
            self._log_rm_images(x, target_rm, pred_rm, step_name)

        # flatten roadmap tensors, convert target rm from True/False to 1/0
        target_rm = target_rm.view(target_rm.size(0), -1).float()
        pred_rm = pred_rm.view(pred_rm.size(0), -1)

        # calculate binary cross entropy loss
        loss = F.binary_cross_entropy(target_rm, pred_rm)

        return loss

    def _log_rm_images(self, x, target_rm, pred_rm, step_name, limit=1):
        # log 6 images stitched wide, target/true roadmap and predicted roadmap
        # take first image in the batch
        x = x[:limit]
        target_rm = target_rm[:limit]
        pred_rm = pred_rm[:limit]

        input_images = torchvision.utils.make_grid(x)
        target_roadmaps = torchvision.utils.make_grid(target_rm)
        pred_roadmaps = torchvision.utils.make_grid(pred_rm)

        self.logger.experiment.add_image(f'{step_name}_input_images', input_images, self.trainer.global_step)
        self.logger.experiment.add_image(f'{step_name}_target_roadmaps', target_roadmaps, self.trainer.global_step)
        self.logger.experiment.add_image(f'{step_name}_pred_roadmaps', pred_roadmaps, self.trainer.global_step)

    def training_step(self, batch, batch_idx):
        loss = self._run_step(batch, batch_idx, step_name='train')
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss = self._run_step(batch, batch_idx, step_name='valid')
        tensorboard_logs = {'val_loss': loss}
        return {'val_loss': loss, 'log': tensorboard_logs}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0005)

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
                            batch_size=self.batch_size,
                            shuffle=True,
                            num_workers=4,
                            collate_fn=collate_fn)
        return loader

    def val_dataloader(self):
        # don't shuffle validation batches
        loader = DataLoader(self.labeled_validset,
                            batch_size=self.batch_size,
                            shuffle=False,
                            num_workers=4,
                            collate_fn=collate_fn)
        return loader

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = HyperOptArgumentParser(parents=[parent_parser], add_help=False)

        # want to optimize this parameter
        parser.opt_list('--batch_size', type=int, default=24, options=[24, 16, 10, 8], tunable=True)
        #parser.add_argument('--batch_size', type=int, default=2)
        # fixed arguments
        parser.add_argument('--link', type=str, default='/Users/annika/Developer/driving-dirty/data')
        parser.add_argument('--checkpoint', type=str, default='/Users/annika/Developer/driving-dirty/lightning_logs/version_3/checkpoints/epoch=4.ckpt')
        return parser

    def patch_checkpoint(self, name):
        from pytorch_lightning.core.saving import ModelIO, load_hparams_from_tags_csv
        import re
        from argparse import Namespace

        d = torch.load(name, map_location=torch.device('cpu'))
        csv_path = name.split('checkpoints')[0]
        hparams = load_hparams_from_tags_csv(f'{csv_path}/meta_tags.csv')
        d['hparams'] = hparams.__dict__
        name = re.sub('epoch=', 'fix_epoch=', name)

        torch.save(d, name)
        return name


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = RoadMap.add_model_specific_args(parser)
    args = parser.parse_args()

    model = RoadMap(args)
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model)

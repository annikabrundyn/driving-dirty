
import os
import random

from argparse import ArgumentParser, Namespace

import numpy as np

import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from pytorch_lightning import LightningModule, Trainer
from test_tube import HyperOptArgumentParser

from data_helper import LabeledDataset, UnlabeledDataset
from helper import collate_fn

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


##################### BASIC AE ############################

class BasicAE(LightningModule):
    def __init__(self, hparams=None):
        super().__init__()
        # attach hparams to log hparams to the loggers (like tensorboard)
        self.__check_hparams(hparams)
        self.hparams = hparams

        self.encoder = self.init_encoder(self.hidden_dim, self.latent_dim,
                                         self.in_channels, self.input_height, self.input_width)
        self.decoder = self.init_decoder(self.hidden_dim, self.latent_dim,
                                         self.in_channels, self.output_height, self.output_width)

    def __check_hparams(self, hparams):
        self.hidden_dim = hparams.hidden_dim if hasattr(hparams, 'hidden_dim') else 128
        self.latent_dim = hparams.latent_dim if hasattr(hparams, 'latent_dim') else 128

        self.input_width = hparams.input_width if hasattr(hparams, 'input_width') else 306*6
        self.input_height = hparams.input_height if hasattr(hparams, 'input_height') else 256

        self.output_width = hparams.output_width if hasattr(hparams, 'output_width') else 306
        self.output_height = hparams.output_height if hasattr(hparams, 'output_height') else 256

        self.batch_size = hparams.batch_size if hasattr(hparams, 'batch_size') else 24
        self.in_channels = hparams.in_channels if hasattr(hparams, 'in_channels') else 3

    def init_encoder(self, hidden_dim, latent_dim, in_channels, input_height, input_width):
        encoder = Encoder(hidden_dim, latent_dim, in_channels, input_height, input_width)
        return encoder

    def init_decoder(self, hidden_dim, latent_dim, in_channels, output_height, output_width):
        decoder = Decoder(hidden_dim, latent_dim, in_channels, output_height, output_width)
        return decoder

    def six_to_one_task(self, x):
        # reorder and stitch images together in wide format
        x = x[:, [0, 1, 2, 5, 4, 3]]
        b, num_imgs, c, h, w = x.size()
        x = x.permute(0, 2, 3, 1, 4).reshape(b, c, h, -1)

        # randomly choose one of the 6 pictures to be blacked out
        target_img_index = np.random.randint(0,5)
        start_i = target_img_index * 306
        end_i = start_i + 306

        y = x[:, :, :, start_i: end_i]
        y = y.clone()

        x[:, :, :, start_i: end_i] = 0.0

        # check that the dimensions are correct
        assert x.size(-1) == 6 * 306
        assert y.size(-1) == 306

        return x, y

    def forward(self, z):
        return self.decoder(z)

    def _run_step(self, batch, batch_idx, step_name):
        x, y = self.six_to_one_task(batch)

        # Encode - z has dim batch_size x latent_dim
        z = self.encoder(x)

        # Decode - y_hat has same dim as true y
        y_hat = self(z)

        if batch_idx % 1000 == 0:
            self._log_images(y, y_hat, step_name)

        # consider replacing this reconstruction loss with something else
        # note - flatten both the true and the predicted to calculated mse loss
        loss = F.mse_loss(y, y_hat)

        return loss

    def _log_images(self, y, y_hat, step_name, limit=1):
        y = y[:limit]
        y_hat = y_hat[:limit]

        pred_images = torchvision.utils.make_grid(y_hat)
        target_images = torchvision.utils.make_grid(y)

        self.logger.experiment.add_image(f'{step_name}_predicted_images', pred_images, self.trainer.global_step)
        self.logger.experiment.add_image(f'{step_name}_target_images', target_images, self.trainer.global_step)

    def training_step(self, batch, batch_idx):
        train_loss = self._run_step(batch, batch_idx, step_name='train')
        train_tensorboard_logs = {'train_loss': train_loss}
        return {'loss': train_loss, 'log': train_tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        val_loss = self._run_step(batch, batch_idx, step_name='valid')
        return {'val_loss': val_loss}

    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_tensorboard_logs = {'avg_val_loss': avg_val_loss}
        return {'val_loss': avg_val_loss, 'log': val_tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0005)

    def prepare_data(self):
        image_folder = self.hparams.link

        transform = torchvision.transforms.ToTensor()

        unlabeled_dataset = UnlabeledDataset(image_folder=image_folder,
                                             scene_index=np.arange(106),
                                             first_dim='sample',
                                             transform=transform)

        trainset_size = round(0.8 * len(unlabeled_dataset))
        validset_size = round(0.2 * len(unlabeled_dataset))

        self.trainset, self.validset = torch.utils.data.random_split(unlabeled_dataset,
                                                                      lengths = [trainset_size,
                                                                                 validset_size])

    def train_dataloader(self):
        loader = torch.utils.data.DataLoader(self.trainset,
                                             batch_size=self.batch_size,
                                             shuffle=True,
                                             num_workers=4)
        return loader

    def val_dataloader(self):
        loader = torch.utils.data.DataLoader(self.trainset,
                                             batch_size=self.batch_size,
                                             shuffle=False,
                                             num_workers=4)
        return loader

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = HyperOptArgumentParser(parents=[parent_parser], add_help=False)
        parser.opt_list('--hidden_dim', type=int, default=128, options=[128, 256, 512], tunable=True,
                            help='itermediate layers dimension before embedding for default encoder/decoder')
        parser.opt_list('--latent_dim', type=int, default=128, options=[64, 128, 256, 512], tunable=True,
                            help='dimension of latent variables z')

        parser.add_argument('--input_width', type=int, default=306*6,
                            help='input image width - 28 for MNIST (must be even)')
        parser.add_argument('--input_height', type=int, default=256,
                            help='input image height - 28 for MNIST (must be even)')
        parser.add_argument('--output_width', type=int, default=306)
        parser.add_argument('--output_height', type=int, default=256)

        parser.opt_list('--batch_size', type=int, default=16, options=[64, 32, 24, 16, 10, 8], tunable=False)
        parser.add_argument('--in_channels', type=int, default=3)

        parser.add_argument('--link', type=str, default='/Users/annika/Developer/driving-dirty/data')
        return parser


##################### ENCODER, DECODER COMPONENTS ############################
class Encoder(torch.nn.Module):
    """
    Takes as input an image, uses a CNN to extract features which
    get split into a mu and sigma vector
    """
    def __init__(self, hidden_dim, latent_dim, in_channels, input_height, input_width):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.input_height = input_height
        self.input_width = input_width
        self.in_channels = in_channels

        self.c1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.c2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.c3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)

        self.pooling_size = 4
        conv_out_dim = self._calculate_output_dim(in_channels, input_height, input_width, self.pooling_size)

        self.fc1 = DenseBlock(conv_out_dim, hidden_dim)
        self.fc2 = DenseBlock(hidden_dim, hidden_dim)

        self.fc_z_out = nn.Linear(hidden_dim, latent_dim)

    def _calculate_output_dim(self, in_channels, input_height, input_width, pooling_size):
        x = torch.rand(1, in_channels, input_height, input_width)
        x = self.c3(self.c2(self.c1(x)))
        x = x.view(-1).unsqueeze(0).unsqueeze(0)
        x = F.max_pool1d(x, kernel_size=pooling_size)
        return x.size(-1)

    def forward(self, x):
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = F.relu(self.c3(x))
        x = x.view(x.size(0), -1).unsqueeze(1)
        x = F.max_pool1d(x, kernel_size=self.pooling_size).squeeze(1)
        x = self.fc1(x)
        x = self.fc2(x)

        z = self.fc_z_out(x)
        return z


class Decoder(torch.nn.Module):
    """
    takes in latent vars and reconstructs an image
    """
    def __init__(self, hidden_dim, latent_dim, in_channels, output_height, output_width):

        super().__init__()

        self.deconv_dim_h, self.deconv_dim_w = self._calculate_output_size(in_channels,
                                                                           output_height,
                                                                           output_width)

        self.latent_dim = latent_dim
        self.fc1 = DenseBlock(latent_dim, hidden_dim)
        self.fc2 = DenseBlock(hidden_dim, self.deconv_dim_h * self.deconv_dim_w * 64)
        self.dc1 = nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1)
        self.dc2 = nn.ConvTranspose2d(32, 32, kernel_size=3, padding=1)
        self.dc3 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.dc4 = nn.ConvTranspose2d(32, in_channels, kernel_size=1, stride=1)

    def _calculate_output_size(self, in_channels, output_height, output_width):
        x = torch.rand(1, in_channels, output_height, output_width)
        dc1 = nn.Conv2d(in_channels, 32, kernel_size=1, stride=1)
        dc2 = nn.Conv2d(32, 32, kernel_size=2, stride=2)
        dc3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        dc4 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        x = dc4(dc3(dc2(dc1(x))))
        return x.size(-2), x.size(-1)

    def forward(self, z):
        x = self.fc1(z)
        x = self.fc2(x)
        x = x.view(x.size(0), 64, self.deconv_dim_h, self.deconv_dim_w)
        x = F.relu(self.dc1(x))
        x = F.relu(self.dc2(x))
        x = F.relu(self.dc3(x))
        x = self.dc4(x)   ### NOTE: removed the F.sigmoid on this layer - colour images
        return x


class DenseBlock(nn.Module):
    def __init__(self, in_dim, out_dim, drop_p=0.2):
        super().__init__()
        self.drop_p = drop_p
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc_bn = nn.BatchNorm1d(out_dim)
        self.in_dim = in_dim

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc_bn(x)
        x = F.relu(x)
        x = F.dropout(x, self.drop_p)
        return x



##################### ROADMAP ############################

class RoadMap(LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.output_dim = 800 * 800
        #self.kernel_size = 4

        # TODO: add pretrained weight path
        # TODO: remove this to train models again
        #d = dict(
        #    latent_dim = 64,
        #    hidden_dim = 128,
        #    batch_size = 16
        #)
        #hparams2 = Namespace(**d)

        # pretrained feature extractor - using our own trained Encoder
        self.ae = BasicAE.load_from_checkpoint(self.hparams.pretrained_path)
        #self.ae = BasicAE(hparams2)
        self.frozen = True
        self.ae.freeze()
        self.ae.decoder = None

        # MLP layers: feature embedding --> predict binary roadmap
        self.fc1 = nn.Linear(self.ae.latent_dim, self.output_dim)
        #self.fc2 = nn.Linear(200000, self.output_dim)

    def wide_stitch_six_images(self, sample):
        # change from tuple len([6 x 3 x H x W]) = b --> tensor [b x 6 x 3 x H x W]
        x = torch.stack(sample, dim=0)

        # reorder order of 6 images (in first dimension) to become 180 degree view
        x = x[:, [0, 1, 2, 5, 4, 3]]

        # rearrange axes and reshape to wide format
        b, num_imgs, c, h, w = x.size()
        x = x.permute(0, 2, 3, 1, 4).reshape(b, c, h, -1)
        #assert x.size(-1) == 6 * 306
        return x

    def forward(self, x):
        # wide stitch the 6 images in sample
        x = self.wide_stitch_six_images(x)

        # note: can call forward(x) with self(x)
        # first find representations using the pretrained encoder
        representations = self.ae.encoder(x)

        # now run through MLP
        y = self.fc1(representations)
        #y = torch.sigmoid(self.fc1(representations))

        # reshape prediction to be tensor with b x 800 x 800
        y = y.reshape(y.size(0), 800, 800)

        return y, torch.sigmoid(y)

    def _run_step(self, batch, batch_idx, step_name):
        sample, target, road_image = batch

        # change target roadmap from tuple len([800 x 800]) = b --> tensor [b x 800 x 800]
        target_rm = torch.stack(road_image, dim=0).float()

        # forward pass to find predicted roadmap
        pred_rm, pred_logit_rm = self(sample)

        # every 10 epochs we look at inputs + predictions
        if batch_idx % self.hparams.output_img_freq == 0:
            x = self.wide_stitch_six_images(sample)
            self._log_rm_images(x, target_rm, pred_logit_rm, step_name)

        # calculate loss between pixels
        # if self.hparams.loss_fn == "mse":
        #loss = F.mse_loss(target_rm, pred_rm)

        # elif self.hparams.loss_fn == "bce":
        # flatten and calculate binary cross entropy
        batch_size = target_rm.size(0)
        target_rm_flat = target_rm.view(batch_size, -1)
        pred_rm_flat = pred_rm.view(batch_size, -1)
        loss = F.binary_cross_entropy_with_logits(pred_rm_flat, target_rm_flat) #ok.

        return loss, target_rm, pred_rm, pred_logit_rm

    def _log_rm_images(self, x, target_rm, pred_rm, step_name, limit=1):
        # log 6 images stitched wide, target/true roadmap and predicted roadmap
        # take first image in the batch
        x = x[:limit]
        target_rm = target_rm[:limit]
        pred_rm = pred_rm[:limit].round()

        input_images = torchvision.utils.make_grid(x)
        target_roadmaps = torchvision.utils.make_grid(target_rm)
        pred_roadmaps = torchvision.utils.make_grid(pred_rm)

        self.logger.experiment.add_image(f'{step_name}_input_images', input_images, self.trainer.global_step)
        self.logger.experiment.add_image(f'{step_name}_target_roadmaps', target_roadmaps, self.trainer.global_step)
        self.logger.experiment.add_image(f'{step_name}_pred_roadmaps', pred_roadmaps, self.trainer.global_step)

    def training_step(self, batch, batch_idx):

        if self.current_epoch >= self.hparams.unfreeze_epoch_no and self.frozen:
            self.frozen=False
            self.ae.unfreeze()

        train_loss, _, _, _  = self._run_step(batch, batch_idx, step_name='train')
        train_tensorboard_logs = {'train_loss': train_loss}
        return {'loss': train_loss, 'log': train_tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        val_loss, target_rm, pred_rm, pred_logit_rm = self._run_step(batch, batch_idx, step_name='valid')

        # calculate threat score
        val_ts = compute_ts_road_map(target_rm, pred_logit_rm)
        val_ts_rounded = compute_ts_road_map(target_rm, pred_logit_rm.round())

        return {'val_loss': val_loss, 'val_ts_rounded': val_ts_rounded, 'val_ts': val_ts}

    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_val_ts = torch.stack([x['val_ts'] for x in outputs]).mean()
        avg_val_ts_rounded = torch.stack([x['val_ts_rounded'] for x in outputs]).mean()
        val_tensorboard_logs = {'avg_val_loss': avg_val_loss,
                                'avg_val_ts_rounded': avg_val_ts_rounded,
                                'avg_val_ts': avg_val_ts}
        return {'val_loss': avg_val_loss, 'log': val_tensorboard_logs}

    def configure_optimizers(self):
        optimizer =  torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=10)
        return [optimizer], [scheduler]

    def prepare_data(self):
        image_folder = self.hparams.link
        annotation_csv = self.hparams.link + '/annotation.csv'
        labeled_scene_index = np.arange(106, 134)
        trainset_size = round(0.8 * len(labeled_scene_index))

        # split into train / validation sets at the scene index level
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
        parser.opt_list('--learning_rate', type=float, default=1e-3, options=[1e-3, 1e-4, 1e-5], tunable=False)
        parser.opt_list('--unfreeze_epoch_no', type=int, default=0, options=[0, 20], tunable=True)

        parser.add_argument('--batch_size', type=int, default=16)
        # fixed arguments
        parser.add_argument('--link', type=str, default='/scratch/ab8690/DLSP20Dataset/data')
        parser.add_argument('--pretrained_path', type=str, default='/scratch/ab8690/logs/space_bb_pretrain/lightning_logs/version_9604234/checkpoints/epoch=23.ckpt')
        parser.add_argument('--output_img_freq', type=int, default=500)
        return parser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = RoadMap.add_model_specific_args(parser)
    args = parser.parse_args()

    model = RoadMap(args)
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model)

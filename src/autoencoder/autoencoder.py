import os
from argparse import ArgumentParser

import torch
import torchvision
from torch.nn import functional as F
from pytorch_lightning import LightningModule, Trainer
from test_tube import HyperOptArgumentParser

import numpy as np

from src.autoencoder.components import Encoder, Decoder
from src.utils.data_helper import UnlabeledDataset


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
        self.latent_dim = hparams.latent_dim if hasattr(hparams, 'latent_dim') else 32

        self.input_width = hparams.input_width if hasattr(hparams, 'input_width') else 306*6
        self.input_height = hparams.input_height if hasattr(hparams, 'input_height') else 256

        self.output_width = hparams.output_width if hasattr(hparams, 'output_width') else 306
        self.output_height = hparams.output_height if hasattr(hparams, 'output_height') else 256

        self.batch_size = hparams.batch_size if hasattr(hparams, 'batch_size') else 2
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
        # x = x.permute(0, 2, 1, 3, 4).reshape(self.batch_size, self.in_channels, self.input_height, -1)

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
        #input, _ = batch

        # (batch, imgs, c, h=256, w=306)
        #input = torch.randn(self.batch_size, 6, self.in_channels, 256, 306)

        x, y = self.six_to_one_task(batch)

        # Encode - z has dim batch_size x latent_dim
        z = self.encoder(x)

        # Decode - y_hat has same dim as true y
        y_hat = self(z)

        if batch_idx % 2 == 0:
            self._log_images(x, y, step_name)

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
        loss = self._run_step(batch, batch_idx, step_name='train')
        tensorboard_logs = {'mse_loss': loss}

        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss = self._run_step(batch, batch_idx, step_name='valid')

        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'mse_loss': avg_loss}

        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

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
        parser.add_argument('--hidden_dim', type=int, default=64,
                            help='itermediate layers dimension before embedding for default encoder/decoder')
        parser.add_argument('--latent_dim', type=int, default=32,
                            help='dimension of latent variables z')

        parser.add_argument('--input_width', type=int, default=306*6,
                            help='input image width - 28 for MNIST (must be even)')
        parser.add_argument('--input_height', type=int, default=256,
                            help='input image height - 28 for MNIST (must be even)')
        parser.add_argument('--output_width', type=int, default=306)
        parser.add_argument('--output_height', type=int, default=256)

        parser.add_argument('--batch_size', type=int, default=24)
        parser.add_argument('--in_channels', type=int, default=3)

        parser.add_argument('--link', type=str, default='/Users/annika/Developer/driving-dirty/data')
        return parser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = BasicAE.add_model_specific_args(parser)
    args = parser.parse_args()

    ae = BasicAE(args)
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(ae)
import os
from argparse import ArgumentParser

import numpy as np
import torch
from pytorch_lightning import LightningModule, Trainer
from torch.nn import functional as F
from components import Encoder, Decoder
from pl_bolts.datamodules import MNISTDataLoaders


class BasicAE(LightningModule):

    def __init__(
            self,
            hparams=None,
    ):
        super().__init__()
        # attach hparams to log hparams to the loggers (like tensorboard)
        self.__check_hparams(hparams)
        self.hparams = hparams

        self.dataloaders = MNISTDataLoaders(save_path=os.getcwd())

        self.encoder = self.init_encoder(self.hidden_dim, self.latent_dim,
                                         self.in_channels, self.input_height, self.input_width)
        self.decoder = self.init_decoder(self.hidden_dim, self.latent_dim,
                                         self.in_channels, self.output_height, self.output_width)

    def __check_hparams(self, hparams):
        self.hidden_dim = hparams.hidden_dim if hasattr(hparams, 'hidden_dim') else 128
        self.latent_dim = hparams.latent_dim if hasattr(hparams, 'latent_dim') else 32
        self.input_width = hparams.input_width if hasattr(hparams, 'input_width') else 306*6
        self.input_height = hparams.input_height if hasattr(hparams, 'input_height') else 256
        self.output_width = hparams.input_width if hasattr(hparams, 'output_width') else 306
        self.output_height = hparams.input_height if hasattr(hparams, 'output_height') else 256
        self.batch_size = hparams.batch_size if hasattr(hparams, 'batch_size') else 32
        self.in_channels = hparams.batch_size if hasattr(hparams, 'in_channels') else 3

    def init_encoder(self, hidden_dim, latent_dim, in_channels, input_height, input_width):
        encoder = Encoder(hidden_dim, latent_dim, in_channels, input_height, input_width)
        return encoder

    def init_decoder(self, hidden_dim, latent_dim, in_channels, output_height, output_width):
        decoder = Decoder(hidden_dim, latent_dim, in_channels, output_height, output_width)
        return decoder

    def six_to_one_task(self, x):
        # stitch images together
        x = x[:, [0, 1, 2, 5, 4, 3]]
        x = x.permute(0, 2, 1, 3, 4).reshape(32, 3, 256, -1)

        # randomly choose one picture to be blacked out - find index wrt wide image
        target_img_index = np.random.randint(0,5)
        start_i = target_img_index * 306
        end_i = start_i + 306

        y = x[:, :, :, start_i: end_i]
        x[:, :, :, start_i: end_i] = 0.0

        assert x.size(-1) == 6 * 306
        assert y.size(-1) == 306

        return x, y

    def forward(self, z):
        return self.decoder(z)

    def _run_step(self, batch):
        input, _ = batch

        # (batch, imgs, c, h=256, w=306)
        input = torch.randn(32, 6, 3, 256, 306)

        x, y = self.six_to_one_task(input)

        z = self.encoder(x)
        x_hat = self(z)

        loss = F.mse_loss(x.view(x.size(0), -1), x_hat)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._run_step(batch)

        tensorboard_logs = {
            'mse_loss': loss,
        }

        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss = self._run_step(batch)

        return {
            'val_loss': loss,
        }

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        tensorboard_logs = {'mse_loss': avg_loss}

        return {
            'avg_val_loss': avg_loss,
            'log': tensorboard_logs
        }

    def test_step(self, batch, batch_idx):
        loss = self._run_step(batch)

        return {
            'test_loss': loss,
        }

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()

        tensorboard_logs = {'mse_loss': avg_loss}

        return {
            'avg_test_loss': avg_loss,
            'log': tensorboard_logs
        }

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def prepare_data(self):
        self.dataloaders.prepare_data()

    def train_dataloader(self):
        return self.dataloaders.train_dataloader(self.batch_size)

    def val_dataloader(self):
        return self.dataloaders.val_dataloader(self.batch_size)

    def test_dataloader(self):
        return self.dataloaders.test_dataloader(self.batch_size)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--hidden_dim', type=int, default=128,
                            help='itermediate layers dimension before embedding for default encoder/decoder')
        parser.add_argument('--latent_dim', type=int, default=32,
                            help='dimension of latent variables z')
        parser.add_argument('--input_width', type=int, default=306*6,
                            help='input image width - 28 for MNIST (must be even)')
        parser.add_argument('--input_height', type=int, default=256,
                            help='input image height - 28 for MNIST (must be even)')
        parser.add_argument('--output_width', type=int, default=306)
        parser.add_argument('--output_height', type=int, default=256)

        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--in_channels', type=int, default=3)
        return parser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = BasicAE.add_model_specific_args(parser)
    args = parser.parse_args()

    ae = BasicAE(args)
    trainer = Trainer()
    trainer.fit(ae)
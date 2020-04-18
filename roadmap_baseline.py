import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
import torchvision.models as models

import pytorch_lightning as pl


class RoadMap(pl.LightningModule):

    def __init__(self):
        super().__init__()

        embedding_dim = 1000
        output_dim = 800*800

        self.feature_extractor = models.resnet50(
                                    pretrained=True,
                                    num_classes=embedding_dim)

        self.feature_extractor.eval()

        # use the pretrained model to predict binary roadmap
        self.linear_1 = nn.Linear(embedding_dim, output_dim)


    def forward(self, x):
        # called with self(x)
        representations = self.feature_extractor(x)
        outputs = F.sigmoid(self.linear_1(representations))
        return outputs

    def training_step(self, batch, batch_idx):
        # REQUIRED
        x, y = batch
        y = torch.round(torch.rand(32, 800*800))
        y_hat = self.forward(x)
        loss = F.binary_cross_entropy(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        y = torch.round(torch.rand(32, 800*800))
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
        loader = DataLoader(self.cifar_train, batch_size=32)
        #labeled_trainset = LabeledDataset(image_folder=image_folder,
        #                                  annotation_file=annotation_csv,
        #                                  scene_index=labeled_scene_index,
        #                                  transform=transform,
        #                                  extra_info=True
        #                                  )
        #loader = DataLoader(labeled_trainset, batch_size=32, shuffle=True, num_workers=2, collate_fn=collate_fn)
        #loader = DataLoader(self.mnist_train, batch_size=32)
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.cifar_test, batch_size=32)
        return loader

    def test_dataloader(self):
        loader = DataLoader(self.cifar_test, batch_size=32)
        return loader


if __name__ == '__main__':
    #parser = ArgumentParser()
    #parser = pl.Trainer.add_argparse_args(parser)
    #parser = VAE.add_model_specific_args(parser)
    #args = parser.parse_args()

    model = RoadMap()
    trainer = pl.Trainer()
    trainer.fit(model)
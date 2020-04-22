import random

from argparse import ArgumentParser
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models

from src.utils.data_helper import LabeledDataset
from src.utils.helper import collate_fn

import pytorch_lightning as pl

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

class RoadMap(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()

        output_dim = 800*800
        self.hparams = hparams

        # pretrained feature extractor
        self.feature_extractor = models.resnet50(
                                    pretrained=True,
                                    num_classes=1000)
        # remove last layer (FC) of resnet by making it the identity
        self.feature_extractor.fc = Identity()
        # put the feature extractor into eval model so that the weights are frozen and dont train
        self.feature_extractor.eval()

        # FC layer to predict - this is the only layer I'm training atm
        self.linear_1 = nn.Linear(512, output_dim)

    def forward(self, x):
        # called with self(x)
        representations = self.feature_extractor(x)
        # import pdb; pdb.set_trace()
        # apply a sigmoid activation to the linear layer
        #outputs = self.max_pool(representations)
        representations = representations.view(self.hparams.batch_size, -1).unsqueeze(1)
        representations = F.max_pool1d(representations, kernel_size=4)
        outputs = F.sigmoid(self.linear_1(representations))
        return outputs

    def _run_step(self, batch, batch_idx):
        # this function is going to be used for one step of the training/validation loops
        # so basically for one batch in one epoch - we take in that batch, predict the outputs, calculate
        # the loss from the predictions and return that loss
        # pytorch lightning is automatically going to update the weights for us - no need to run explicitly
        sample, target, road_image = batch

        # change dim from tuple with length(tuple) = batch_size containing tensors with size [6 x 3 x H x W]
        # --> to tensor with size [batch_size x 6 x 3 x H x W]
        x = torch.stack(sample, dim=0)

        # reorder 6 images for each sample so they're sequential order for the wide view
        x = x[:, [0, 1, 2, 5, 4, 3]]

        # reshape to wide format - stitch 6 images side by side
        x = x.permute(0, 2, 1, 3, 4).reshape(self.hparams.batch_size, 3, 256, -1)

        # get the road-image y with shape [batch_size x 800 x 800]
        y = torch.stack(road_image, dim=0)

        # forward pass to calculate outputs
        outputs = self(x)

        # flatten y and outputs in order to run binary cross entropy fn --> shape [batch_size x 800*800]
        # also convert true y from True/False to 1/0
        outputs = outputs.view(outputs.size(0), -1)
        y = y.view(y.size(0), -1).float()

        loss = F.binary_cross_entropy(outputs, y)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._run_step(batch, batch_idx)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss = self._run_step(batch, batch_idx)
        tensorboard_logs = {'val_loss': loss}
        return {'val_loss': loss, 'log': tensorboard_logs}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        loss = self._run_step(batch, batch_idx)
        tensorboard_logs = {'test_loss': loss}
        return {'test_loss': loss, 'log': tensorboard_logs}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss}
        return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def prepare_data(self):

	# here we download and transform the data but don't load them into dataloaders yet
        # the dataloaders are run batch by batch where this is run fully and once before beginning training
        image_folder = self.hparams.link #'/scratch/ab8690/DLSP20Dataset/data' #i had to update
        annotation_csv = self.hparams.link + '/annotation.csv' #'/scratch/ab8690/DLSP20Dataset/data/annotation.csv'

        # split into train and validation - did this using scene indices but not sure if we want to split the
        # datasets at the scene folder level or at the sample level - could try both
        np.random.shuffle(labeled_scene_index)
        training_set_index = labeled_scene_index[:24]
        validation_set_index = labeled_scene_index[24:]

        transform = transforms.ToTensor()

        # training set
        self.labeled_trainset = LabeledDataset(image_folder=image_folder,
                                          annotation_file=annotation_csv,
                                          scene_index=training_set_index,
                                          transform=transform,
                                          extra_info=False
                                          )
        # validation set
        self.labeled_validset = LabeledDataset(image_folder=image_folder,
                                          annotation_file=annotation_csv,
                                          scene_index=validation_set_index,
                                          transform=transform,
                                          extra_info=False
                                          )

    def train_dataloader(self):
        loader = DataLoader(self.labeled_trainset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=4,
                            collate_fn=collate_fn)
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.labeled_validset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=4,
                            collate_fn=collate_fn)
        return loader

    def test_dataloader(self):
        pass
        #loader = DataLoader(self.cifar_test, batch_size=batch_size)
        #return loader


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


if __name__ == '__main__':
    parser = ArgumentParser()
    #parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--link', type=str, default='/Users/annika/Developer/driving-dirty/data')
    #parser = VAE.add_model_specific_args(parser)
    args = parser.parse_args()

    unlabeled_scene_index = np.arange(106)
    labeled_scene_index = np.arange(106, 134)

    model = RoadMap(args)
    trainer = pl.Trainer(gpus=1)
    trainer.fit(model)

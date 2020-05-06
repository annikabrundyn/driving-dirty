import random
import numpy as np
import torch

from argparse import ArgumentParser, Namespace

import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from pytorch_lightning import LightningModule, Trainer
from test_tube import HyperOptArgumentParser

from src.utils.data_helper import LabeledDataset
from src.utils.helper import collate_fn, boxes_to_binary_map, compute_ts_road_map
from src.autoencoder.autoencoder import BasicAE
from src.bounding_box_model.spatial_bb.components import SpatialMappingCNN, RoadMapBoxesMergingCNN

random.seed(20200505)
np.random.seed(20200505)
torch.manual_seed(20200505)


class BBSpatialRoadMap(LightningModule):

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

        self.space_map_cnn = SpatialMappingCNN()

        self.box_merge = RoadMapBoxesMergingCNN()

    def wide_stitch_six_images(self, x):
        # change from tuple len([6 x 3 x H x W]) = b --> tensor [b x 6 x 3 x H x W]
        #x = torch.stack(sample, dim=0)

        # reorder order of 6 images (in first dimension) to become 180 degree view
        x = x[:, [0, 1, 2, 5, 4, 3]]

        # rearrange axes and reshape to wide format
        b, num_imgs, c, h, w = x.size()
        x = x.permute(0, 2, 3, 1, 4).reshape(b, c, h, -1)
        #assert x.size(-1) == 6 * 306
        return x

    def forward(self, x, rm):
        # spatial representation
        import pdb; pdb.set_trace()
        spacial_rep = self.space_map_cnn(x)

        # selfsupervised representation
        x = self.wide_stitch_six_images(x)
        ssr = self.ae.encoder(x, c3_only=True)

        # combine two -> [b, 800, 800]
        yhat = self.box_merge(ssr, spacial_rep, rm)
        yhat = yhat.squeeze(1)

        return yhat

    def bb_coord_to_map(self, target):
        # target is tuple with len b
        results = []
        for i, sample in enumerate(target):
            # tuple of len 2 -> [num_boxes, 2, 4]
            sample = sample['bounding_box']
            map = boxes_to_binary_map(sample)
            results.append(map)

        results = torch.tensor(results)
        return results

    def _run_step(self, batch, batch_idx, step_name):
        sample, target, road_image = batch

        # change target from dict of bounding box coords --> [b, 800, 800]
        target_bb_img = self.bb_coord_to_map(target)
        target_bb_img = target_bb_img.type_as(sample[0])

        # change from tuple len([6 x 3 x H x W]) = b --> tensor [b x 6 x 3 x H x W]
        sample = torch.stack(sample, dim=0)
        sample = sample.type_as(sample[0])

        # change input rm from tuple of len b -> [b, 800, 800] -> [b, 1, 800, 800]
        rm = torch.stack(road_image, dim=0).float()
        rm = rm.unsqueeze(1)

        # forward pass to predict
        pred_bb_img = self(sample, rm)

        # every 10 epochs we look at inputs + predictions
        if batch_idx % self.hparams.output_img_freq == 0:
            x0 = sample[0]
            target_bb_img0 = 1 - target_bb_img[0]
            pred_bb_img0 = 1- pred_bb_img[0]

            self._log_rm_images(x0, target_bb_img0, pred_bb_img0, step_name)

        # calculate mse loss between pixels
        batch_size = target_bb_img.size(0)
        target_bb_img = target_bb_img.view(batch_size, -1)
        pred_bb_img = pred_bb_img.view(batch_size, -1)

        if self.hparams.mse_loss:
            loss = F.mse_loss(pred_bb_img, target_bb_img)
        else:
            loss = F.binary_cross_entropy(pred_bb_img, target_bb_img)

        return loss, target_bb_img, pred_bb_img

    def _log_rm_images(self, x, target, pred, step_name, limit=1):

        input_images = torchvision.utils.make_grid(x, normalize=True)
        target = torchvision.utils.make_grid(target, normalize=True)
        pred = torchvision.utils.make_grid(pred, normalize=True)

        self.logger.experiment.add_image(f'{step_name}_input_images', input_images, self.trainer.global_step)
        self.logger.experiment.add_image(f'{step_name}_target_bbs', target, self.trainer.global_step)
        self.logger.experiment.add_image(f'{step_name}_pred_bbs', pred, self.trainer.global_step)

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
        # fixed arguments
        parser.add_argument('--link', type=str, default='/scratch/ab8690/DLSP20Dataset/data')
        parser.add_argument('--pretrained_path', type=str, default='/scratch/ab8690/logs/dd_pretrain_ae/lightning_logs/version_9234267/checkpoints/epoch=42.ckpt')
        parser.add_argument('--output_img_freq', type=int, default=500)
        parser.add_argument('--unfreeze_epoch_no', type=int, default=20)

        parser.add_argument('--mse_loss', default=False, action='store_true')
        return parser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = BBSpatialRoadMap.add_model_specific_args(parser)
    args = parser.parse_args()

    model = BBSpatialRoadMap(args)
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model)

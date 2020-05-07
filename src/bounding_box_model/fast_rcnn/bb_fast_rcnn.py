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
from src.utils.helper import collate_fn, boxes_to_binary_map, compute_ts_road_map, log_fast_rcnn_images
from src.autoencoder.autoencoder import BasicAE
from src.bounding_box_model.spatial_bb.components import SpatialMappingCNN, RoadMapBoxesMergingCNN

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

import matplotlib.pyplot as plt

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
        d = dict(
            latent_dim = 64,
            hidden_dim = 128,
            batch_size = 16
        )
        hparams2 = Namespace(**d)

        # ------------------
        # PRE-TRAINED MODEL
        # ------------------
        #self.backbone = BasicAE.load_from_checkpoint(self.hparams.pretrained_path)
        ae = BasicAE(hparams2)
        ae.freeze()
        self.backbone = ae.encoder
        self.backbone.c3_only = True
        self.backbone.out_channels = 32

        # ------------------
        # FAST RCNN
        # ------------------
        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                           aspect_ratios=((0.5, 1.0, 2.0),))

        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                        output_size=7,
                                                        sampling_ratio=2)
        self.fast_rcnn = FasterRCNN(
            self.backbone,
            num_classes=9,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler
        )

        # for unfreezing encoder later
        self.frozen = True

    def wide_stitch_six_images(self, x):
        # change from tuple len([6 x 3 x H x W]) = b --> tensor [b x 6 x 3 x H x W]
        #x = torch.stack(sample, dim=0)

        # reorder order of 6 images (in first dimension) to become 180 degree view
        x = x[:, [0, 1, 2, 5, 4, 3]]

        # rearrange axes and reshape to wide format
        b, num_imgs, c, h, w = x.size()
        x = x.permute(0, 2, 3, 1, 4).reshape(b, c, h, -1)
        return x

    def forward(self, ssr, targets):
        losses_dict = self.fast_rcnn(ssr, targets)
        return losses_dict

    def _run_step(self, batch, batch_idx, step_name):
        # images, target and roadimage are tuples
        images, target, road_image = batch

        # 6 images to 1 long one
        images = torch.stack(images, dim=0)
        images = self.wide_stitch_six_images(images)

        # adjust format for FastRCNN
        images, target = self._format_for_fastrcnn(images, target)

        # aggregate losses
        losses = self(images, target)

        # log images
        #if batch_idx % self.hparams.output_img_freq == 0:


        # in training, the output is a dict of scalars
        if step_name == 'train':
            loss_classifier = losses['loss_classifier'].double()
            loss_box_reg = losses['loss_box_reg'].double()
            loss_objectness = losses['loss_objectness'].double()
            loss_rpn_box_reg = losses['loss_rpn_box_reg'].double()
            loss = loss_classifier + loss_box_reg + loss_objectness + loss_rpn_box_reg
            return loss, loss_classifier, loss_box_reg, loss_objectness, loss_rpn_box_reg
        else:
            # in val, the output is a dic of boxes and losses
            # TODO: get boxes - transform corner coordinates -> 4 coordinates for original plot function
            loss = []
            for d in losses:
                loss.append(d['scores'])
            loss = torch.stack(loss).mean()

            # ----------------------
            # LOG VALIDATION IMAGES
            # ----------------------
            if batch_idx % self.hparams.output_img_freq == 0:
                import pdb; pdb.set_trace()
                ### --- log one validation predicted image ---
                # [N, 4]
                predicted_coords_0 = losses[0]['boxes']
                # transform [N, 4] -> [N, 2, 4]
                predicted_coords_0 = self._change_to_old_coord_sys(predicted_coords_0)
                # [N]
                pred_categories_0 = losses[0]['labels']

                target_coords_0 = target[0]['boxes']
                target_coords_0 = self._change_to_old_coord_sys(target_coords_0)
                target_categories_0 = target[0]['labels']

                log_fast_rcnn_images(self, images[0], predicted_coords_0, pred_categories_0,
                                     target_coords_0, target_categories_0,
                                     road_image[0],
                                     step_name)

            return loss, None, None, None, None

    def _change_to_old_coord_sys(self, boxes):
        # boxes dim: [N, 4]
        x_1 = x_3 = boxes[:, 0]
        y_1 = y_4 = boxes[:, 1]
        x_2 = x_4 = boxes[:, 2]
        y_2 = y_3 = boxes[:, 3]

        x_coords = torch.stack([x_2, x_4, x_1, x_3], dim=1)
        y_coords = torch.stack([y_2, y_4, y_1, y_3], dim=1)
        old_coords = torch.stack([x_coords, y_coords], dim=1)

        # old_coords: [N, 2, 4]
        return old_coords

    def _change_coord_sys(self, boxes):
        # boxes dim: [N, 2, 4]
        max_x = boxes[:, 0].max(dim=1)[0]
        min_x = boxes[:, 0].min(dim=1)[0]
        max_y = boxes[:, 1].max(dim=1)[0]
        min_y = boxes[:, 1].min(dim=1)[0]

        # output dim: [N, 4] where each box has [x1, x2, x3, x4]
        coords = torch.stack([min_x, min_y, max_x, max_y], dim=1)
        return coords

    def _format_for_fastrcnn(self, images, target):
        # split batch into list of single images
        # [b, 3, 256, 1836] --> list of length b with elements [3, 256, 1836]
        images = list(image.float() for image in images)

        target = [{k: v for k, v in t.items()} for t in target]
        for d in target:
            d['boxes'] = d.pop('bounding_box')
            d['labels'] = d.pop('category')

            # Change coords to (x0, y0, x1, y1) ie top left and bottom right corners
            # TODO: verify
            d['boxes'] = self._change_coord_sys(d['boxes']).float()
            #num_boxes = d['boxes'].size(0)
            #d['boxes'] = d['boxes'][:, :, [0, -1]].reshape(num_boxes, -1).float()

        return images, target

    def training_step(self, batch, batch_idx):

        if self.current_epoch >= self.hparams.unfreeze_epoch_no and self.frozen:
            self.frozen=False
            #self.backbone.train()

        train_loss, loss_classifier, loss_box_reg, loss_objectness, loss_rpn_box_reg = self._run_step(batch,
                                                                                                batch_idx,
                                                                                                step_name='train')
        train_tensorboard_logs = {'train_loss': train_loss,
                                  'train_loss_classifier': loss_classifier,
                                  'train_loss_box_reg': loss_box_reg,
                                  'train_loss_objectness': loss_objectness,
                                  'train_loss_rpn_box_reg': loss_rpn_box_reg}

        return {'loss': train_loss, 'log': train_tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        val_loss, _, _, _, _ = self._run_step(batch, batch_idx, step_name='valid')

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
        parser.opt_list('--learning_rate', type=float, default=0.001, options=[1e-3, 1e-4, 1e-5], tunable=False)
        parser.add_argument('--batch_size', type=int, default=16)
        # fixed arguments
        parser.add_argument('--link', type=str, default='/scratch/ab8690/DLSP20Dataset/data')
        parser.add_argument('--pretrained_path', type=str, default='/scratch/ab8690/logs/space_bb_pretrain/lightning_logs/version_9604234/checkpoints/epoch=23.ckpt')
        parser.add_argument('--output_img_freq', type=int, default=5)
        parser.add_argument('--unfreeze_epoch_no', type=int, default=0)

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

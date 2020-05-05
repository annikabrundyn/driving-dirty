import os
import random
import sys

from argparse import ArgumentParser
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
import torchvision.models as models

from data_helper import UnlabeledDataset, LabeledDataset
from helper import collate_fn, draw_box

import pytorch_lightning as pl

random.seed(0)


class AutoBlock(nn.Module):
    """
    Auto encoder block which bottlnecks an image and reconstructs it at the same size.
    
    Basic idea: convolution batchnorm activation downsample x2
    """
    def __init__(self,in_ch=3):
        super(AutoBlock,self).__init__()
        self.conv1 = nn.Conv2d(in_ch,32,kernel_size = 5,stride = 1,padding = 2) 
        self.bn1 = nn.BatchNorm2d(32)
        self.downsample1 = nn.Conv2d(32,64,kernel_size = 5,stride = 2,padding = 0)
        self.bn2 = nn.BatchNorm2d(64) 
        self.conv2 = nn.Conv2d(64,64,kernel_size = 5,stride = 1,padding = 2) 
        self.bn3 = nn.BatchNorm2d(64)
        self.downsample2 = nn.Conv2d(64,128,kernel_size = 5,stride = 2,padding = 0)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128,128,kernel_size = 5,stride = 1,padding = 2) 
        self.bn5 = nn.BatchNorm2d(128)
        self.downsample3 = nn.Conv2d(128,256,kernel_size = 5,stride = 2,padding = 0)
        self.bn6 = nn.BatchNorm2d(256)
        self.bottleneck = nn.Conv2d(256,256, kernel_size = 3, stride = 1, padding = 1)
        self.dropout = nn.Dropout2d(p=0.4)
        self.upsample1 = nn.ConvTranspose2d(256,128,kernel_size=5,stride=2,output_padding=1)
        self.bn7 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128,128,kernel_size = 5,stride = 1,padding = 2)  #samme as covn 3! Tie weights?
        self.bn8 = nn.BatchNorm2d(128)
        self.upsample2 = nn.ConvTranspose2d(128,64,kernel_size=5,stride=2,output_padding=1)
        self.bn9 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64,64,kernel_size=5, stride = 1, padding = 2)
        self.bn10 = nn.BatchNorm2d(64)
        self.upsample3 = nn.ConvTranspose2d(64,32,kernel_size=5,stride=2,output_padding=1)
        self.bn11 = nn.BatchNorm2d(32)
        self.conv6 = nn.Conv2d(32,32,kernel_size=5,stride = 1, padding = 2)
        self.bn12 = nn.BatchNorm2d(32)
        
        
        self.output_layer = nn.Conv2d(32,3,kernel_size=1)
        
        
    def forward(self,x):
        x = F.relu(self.bn2(self.downsample1(F.relu(self.bn1(self.conv1(x)))))) #modular
        x = F.relu(self.bn4(self.downsample2(F.relu(self.bn3(self.conv2(x))))))
        x = F.relu(self.bn6(self.downsample3(F.relu(self.bn5(self.conv3(x))))))
        z = self.dropout(F.relu(self.bottleneck(x)))
        x = F.relu(self.bn8(self.conv4(F.relu(self.bn7(self.upsample1(z))))))
        x = F.relu(self.bn10(self.conv5(F.relu(self.bn9(self.upsample2(x))))))
        x = F.relu(self.bn12(self.conv6(F.relu(self.bn11(self.upsample3(x))))))
        x = self.output_layer(x) 
        
        return F.pad(x,(-1,-1,-2,-2)), z
        
        


class Self_Driving_Autoencoder(pl.LightningModule):
    
    def __init__(self,block,in_ch):
        super(block,self).__init__()
        
        self.AE1 = block(in_ch)
        self.AE2 = block(in_ch)
        self.AE3 = block(in_ch)
        self.AE4 = block(in_ch)
        self.AE5 = block(in_ch)
        self.AE6 = block(in_ch)

    
    def forward(self,x):
        xa, za = self.AE1(x[:,0])
        xb, zb = self.AE2(x[:,1])
        xc, zc = self.AE3(x[:,2])
        xd, zd = self.AE4(x[:,3])
        xe, ze = self.AE5(x[:,4])
        xf, zf = self.AE6(x[:,5])

        return torch.stack([xa,xb,xc,xd,xe,xf],dim=1), torch.stack([za,zb,zc,zd,ze,zf],dim=1)
            #immediately gives the same shape as the sample from train loader. 
    
    def _run_step(self, batch, batch_idx):
        # this function is going to be used for one step of the training/validation loops
        # so basically for one batch in one epoch - we take in that batch, predict the outputs, calculate
        # the loss from the predictions and return that loss
        # pytorch lightning is automatically going to update the weights for us - no need to run explicitly
        sample, target, road_image = batch

        # change dim from tuple with length(tuple) = batch_size containing tensors with size [6 x 3 x H x W]
        # --> to tensor with size [batch_size x 6 x 3 x H x W]
        try:
            x = torch.stack(sample, dim=0)
        except:
            x = sample #should already be stacked, if unlabeled dataset. 

        outputs = self(x)

#        x.shape == (3,28,28)       #xs is a tuple of (x,x)
# xs.stack(dim=0) =(2,3,28x28) xs.stack(dim=1) => (3,2,28,28) 

        loss = F.mse_loss(outputs, x) #smooth_l1loss #better for cv,, blurry average of a set of 
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
        
        # the dataloaders are run batch by batch where this is run fully and once before beginning training
        image_folder = '../dat/data' #'/scratch/ab8690/DLSP20Dataset/data' #i had to update
        annotation_csv = image_folder + '/annotation.csv' #'/scratch/ab8690/DLSP20Dataset/data/annotation.csv'

        # split into train and validation - did this using scene indices but not sure if we want to split the
        # datasets at the scene folder level or at the sample level - could try both
        np.random.shuffle(unlabeled_scene_index)
        training_set_index = unlabeled_scene_index[:89]
        validation_set_index = unlabeled_scene_index[89:]

        transform = transforms.ToTensor()

        # training set
        self.unlabeled_trainset = UnlabeledDataset(image_folder=image_folder,
                                          annotation_file=annotation_csv,
                                          scene_index=training_set_index,
                                          transform=transform,
                                          extra_info=False
                                          )
        # validation set
        self.unlabeled_validset = UnlabeledDataset(image_folder=image_folder,
                                          annotation_file=annotation_csv,
                                          scene_index=validation_set_index,
                                          transform=transform,
                                          extra_info=False
                                          )

    def train_dataloader(self):
        loader = DataLoader(self.unlabeled_trainset, batch_size=4, shuffle=True, num_workers=4,
                            collate_fn=collate_fn)
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.unlabeled_validset, batch_size=4, shuffle=True, num_workers=4,
                            collate_fn=collate_fn)
        return loader

    def test_dataloader(self):
        pass
        #loader = DataLoader(self.cifar_test, batch_size=batch_size)
        #return loader
        

if __name__ == '__main__':
    parser = ArgumentParser()
    #parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--batch_size', type=int, default=32)
    #parser.add_argument('--link', type=str, default='/scratch/ab8690/DLSP20Dataset/data')
    #parser = VAE.add_model_specific_args(parser)

    unlabeled_scene_index = np.arange(106)
    labeled_scene_index = np.arange(106, 134)

    model = Self_Driving_Autoencoder(AutoBlock,in_ch=3)
    trainer = pl.Trainer(gpus=1)
    trainer.fit(model)


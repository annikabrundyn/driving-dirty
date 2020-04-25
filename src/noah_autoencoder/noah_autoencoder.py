"""
Dimensionality issues. Grr
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

import os
import random
import sys
import numpy as np
import pandas as pd

from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from src.utils.data_helper import UnlabeledDataset

from src.utils.helper import collate_fn, draw_box


random.seed(0)


class InceptionE(nn.Module):  # inception module
    def __init__(self, in_ch, out_ch,conv_out=3, pool_out=3):
        super(InceptionE, self).__init__()
        self.branch1x1 = BasicConv2d(in_ch, conv_out, kernel_size=1,stride=1)
        self.branch3x3 = BasicConv2d(in_ch, conv_out, kernel_size=3, padding=1,stride=1)
        self.branch5x5 = BasicConv2d(in_ch, conv_out, kernel_size=5, padding=2,stride=1)
        self.branch_pool = BasicConv2d(in_ch, pool_out, kernel_size=1,stride=1)

        self.conv_out = BasicConv2d(4*conv_out,out_ch,kernel_size=1,stride=1)
    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(F.avg_pool3d(x, kernel_size=3, stride=1, padding=1))

        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]

        return self.conv_out(torch.cat(outputs, 1))
    
class BasicConv2d(nn.Module):
    def __init__(self, in_ch, out_ch,kernel_size=3,stride=2,padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch,kernel_size=kernel_size,stride=stride,padding=padding)
        self.batchnorm = nn.BatchNorm2d(out_ch, track_running_stats=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)

        return F.relu(x, inplace=True)
    
class Encoder(nn.Module):
    def __init__(self,in_ch,base_ch,inception_block,basic_block):
        super(Encoder,self).__init__()
        
        self.layer1 = inception_block(in_ch,out_ch=base_ch)
        self.downsample1 = basic_block(base_ch,2*base_ch)
        self.layer2 = inception_block(in_ch=2*base_ch,out_ch = 2*base_ch)
        self.downsample2 = basic_block(2*base_ch,2*2*base_ch)
    def forward(self,x):
        x = self.layer1(x)
        x = self.downsample1(x)
        x = self.layer2(x)
        x = self.downsample2(x)
       
        return x
        
class Decoder(nn.Module):
    """
    Takes in latent vars and reconstructs an image
    """
    def __init__(self,inception_block,in_ch):
        super(Decoder,self).__init__()
        self.upsample1 = nn.ConvTranspose2d(in_ch,in_ch//2,kernel_size=3,stride=2)
        self.layer1 = inception_block(in_ch = in_ch//2,out_ch=in_ch//2)
        self.upsample2 = nn.ConvTranspose2d(in_ch//2,in_ch//4,kernel_size=3,stride=2)
        self.layer2 = inception_block(in_ch = in_ch//4,out_ch=in_ch//4)
        self.output_layer = nn.Conv2d(in_channels=in_ch//4,out_channels=3,kernel_size=1)
    def forward(self,x):
        x = self.upsample1(x)
        x = self.layer1(x)
        x = self.upsample2(x)
        x = self.layer2(x)
        return self.output_layer(x)
    
    
class Inception_Autoencoder(nn.Module):
    def __init__(self,Encoder,Decoder,InceptionBlock,BasicBlock,in_ch,base_ch):
        super(Inception_Autoencoder,self).__init__()
        
        self.Encoder = Encoder(in_ch=in_ch,base_ch=base_ch,inception_block = InceptionBlock,basic_block = BasicBlock)
        self.Decoder = Decoder(inception_block=InceptionBlock,in_ch = 2*2*base_ch) #number of downsamples x 2 is input to decoder. 
        
    def forward(self,x):
        z = self.Encoder(x)
        x = self.Decoder(z)
        
        return x,z
    

class Car_Autoencoder(nn.Module):
    
    def __init__(self,AutoEncoder,Encoder,Decoder,InceptionBlock,BasicBlock,in_ch,base_ch):
        super(Car_Autoencoder,self).__init__()
        #self.prepare_data()
        self.AE = AutoEncoder(Encoder,Decoder,InceptionE,BasicConv2d,in_ch,base_ch)
        
    def forward(self,x):
    
        x,z = self.AE(x[:,3])
        
        return x,z
"""  
    def _run_step(self, batch, batch_idx):
        # this function is going to be used for one step of the training/validation loops
        # so basically for one batch in one epoch - we take in that batch, predict the outputs, calculate
        # the loss from the predictions and return that loss
        # pytorch lightning is automatically going to update the weights for us - no need to run explicitly
        
        sample   = batch #should be for unlabeled
        x = sample #torch.stack(sample, dim=0)
        
        #  BATCH SIZE, 6, 3, 256, 306
        #print("X = ", x)
        #print(x.shape) 
        outputs, z = self(x)  
        target = F.pad(x[:,3],(-2,-1,-1,0))
        loss = F.smooth_l1_loss(outputs, target)
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


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def prepare_data(self):
        
        # the dataloaders are run batch by batch where this is run fully and once before beginning training
        #image_folder = '/scratch/nsk367/pytorch-use/DLSP20/dat/data' 
        image_folder = '/Users/noahkasmanoff/Desktop/Deep_Learning/car/dat/data'
        annotation_csv = image_folder + '/annotation.csv' #'/scratch/ab8690/DLSP20Dataset/data/annotation.csv'

        # split into train and validation - did this using scene indices but not sure if we want to split the
        # datasets at the scene folder level or at the sample level - could try both
        np.random.shuffle(unlabeled_scene_index)
        training_set_index = unlabeled_scene_index[:4]
        validation_set_index = unlabeled_scene_index[4:5]

        transform = transforms.ToTensor()

        # training set
        self.unlabeled_trainset = UnlabeledDataset(image_folder=image_folder, scene_index=training_set_index, first_dim='sample', transform=transform)
        # validation set
        self.unlabeled_validset = UnlabeledDataset(image_folder=image_folder, scene_index=validation_set_index, first_dim='sample', transform=transform)
        
    def train_dataloader(self):
        loader = DataLoader(self.unlabeled_trainset, batch_size=4, shuffle=True, num_workers=4)
        return loader
 
    def val_dataloader(self):
        loader = DataLoader(self.unlabeled_validset, batch_size=4, shuffle=True, num_workers=4)
        return loader

    #def test_dataloader(self):
        #pass
        #loader = DataLoader(self.cifar_test, batch_size=batch_size)
        #return loader
        

if __name__ == '__main__':


    unlabeled_scene_index = np.arange(106)
    
    model = Car_Autoencoder(Inception_Autoencoder,Encoder,Decoder,InceptionE,BasicConv2d,in_ch=3,base_ch=16)
    trainer = pl.Trainer(gpus=0)
    trainer.fit(model)
"""
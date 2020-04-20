"""

Where Noah putting his first bad models. 

"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def conv3x3(inplane, outplane, stride=1,padding=0):
    """
    Simple 3x3x3 convolutional block. Maybe I'll update it's name and break everything :-) 
    """
    return nn.Conv2d(inplane,outplane,kernel_size=3,stride=stride,padding=padding,bias=True)

class BasicBlock(nn.Module):
    """
    Basic convolutional block used for all non-dense blocks in my network. Specifically, this is the downsampling layer and bottleneck layers. 
    Since the bottleneck layer is better suited for dropout, I include the optional choice for either here, with a preset dropout val of 0.5.
    """
    def __init__(self,inplane,outplane,stride = 1,padding = 0, batchnorm=True,dropout=False):
        super(BasicBlock, self).__init__()
        self.padding = padding
        self.conv1 = conv3x3(inplane,outplane,padding=padding,stride=stride)
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.bn1 = nn.BatchNorm2d(outplane)
        self.dropout1 = nn.Dropout2d(p=0.5)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        out = self.conv1(x)
        out = self.relu(out)
        if self.batchnorm:
            out = self.bn1(out)
        if self.dropout:
            out = self.dropout1(out)
        
        return out


class Custom_Autoencoder_Block(nn.Module):
    """
       Fairly custom-izable architecture for an autoencoder which accepts a sample image and combines at the bottleneck. 
    """
    
    def _make_layer(self,block,inplanes,outplanes,blocks,stride=1, padding=1,batchnorm=True,dropout=False): 
        #Make convolutional layer at each stage of U-Net.
        layers = []
        for i in range(0,blocks):
            layers.append(block(inplanes,outplanes,stride,padding,batchnorm,dropout))
            inplanes = outplanes  #how channelling is dealt with. Neat!
        return nn.Sequential(*layers)

    def __init__(self, block,in_ch,ch=32,nblocks = 2):
        super(Custom_Autoencoder_Block,self).__init__()
        #ENCODING
        self.layer1 = self._make_layer(block, in_ch, ch, blocks=nblocks,stride=1) #
        self.downsample1= self._make_layer(block,ch,2*ch, blocks=1,stride=2) #go down like this..
        self.layer2 = self._make_layer(block,2*ch,2*ch,blocks=nblocks,stride=1) #cat with deconv_batchnorm1
        self.downsample2 = self._make_layer(block,2*ch,2*2*ch,blocks=1,stride=2)
        self.layer3 = self._make_layer(block,2*2*ch,2*2*ch,blocks=nblocks,stride=1) #      
        self.downsample3 = self._make_layer(block,2*2*ch,2*2*2*ch,blocks=1,stride=2)
        self.upsample3 = nn.ConvTranspose2d(2*2*2*ch,2*2*ch,3,stride=2,padding=0,output_padding=0)
        self.deconv_batchnorm3 = nn.BatchNorm2d(num_features = 2*2*ch,momentum=0.1)
        self.layer4 = self._make_layer(block,2*2*ch,2*2*ch,blocks=nblocks,stride=1)
        self.upsample2 = nn.ConvTranspose2d(2*2*ch,2*ch,3,stride=2,padding=0,output_padding=0,)
        self.deconv_batchnorm2 = nn.BatchNorm2d(num_features = 2*ch,momentum=0.1) 
        self.layer5 = self._make_layer(block,2*ch,2*ch,blocks=nblocks,stride=1)
        self.upsample1 = nn.ConvTranspose2d(2*ch,ch,3,stride=2,padding=0,output_padding=1,)
        self.deconv_batchnorm1 = nn.BatchNorm2d(num_features = ch,momentum=0.1) 
        self.layer6 =  self._make_layer(block,ch,in_ch,blocks=nblocks,stride=1)
        
    def forward(self,x):
        
        x = self.layer1(x) #x1 is equal to the input after the first layer
        x  = self.downsample1(x) #x is the output of first layer fed into second layer
        x = self.layer2(x) #x2 is output of x fed into third layer
        x  = self.downsample2(x) #x is fourth layer's output from fed in x2
        x  = self.layer3(x) # output of layer 4 fed into layer 5
        z = self.downsample3(x) 
        x  = nn.functional.relu(self.deconv_batchnorm3(self.upsample3(z)),inplace=True)
        x = self.layer4(x)
        x = nn.functional.relu(self.deconv_batchnorm2(self.upsample2(x)),inplace=True)
        x = self.layer5(x)
        x =  nn.functional.relu(self.deconv_batchnorm1(self.upsample1(x)),inplace=True)
        x = self.layer6(x)
        x = F.pad(x,(-7,-7,-4,-4))
        return x,z
    
    
class Custom_Autoencoder_Full(nn.Module):
    """
       Joins 6 separate autoencoder blocks, to get the full context of the image. 
    """
    def __init__(self, block, auto_module ,in_ch,ch=32,nblocks = 2):
        super(Custom_Autoencoder_Full,self).__init__()

        self.module_a = auto_module(block,in_ch,ch,nblocks)
        self.module_b = auto_module(block,in_ch,ch,nblocks)
        self.module_c = auto_module(block,in_ch,ch,nblocks)
        self.module_d = auto_module(block,in_ch,ch,nblocks)
        self.module_e = auto_module(block,in_ch,ch,nblocks)
        self.module_f = auto_module(block,in_ch,ch,nblocks)

    def forward(self,x):
        xa = x[:,0]
        xb = x[:,1]
        xc = x[:,2]
        xd = x[:,3]
        xe = x[:,4]
        xf = x[:,5]
        
        xa, za = self.module_a(xa)
        xb, zb = self.module_b(xb)
        xc, zc = self.module_c(xc)
        xd, zd = self.module_d(xd)
        xe, ze = self.module_e(xe)
        xf, zf = self.module_f(xf)

        return torch.stack([xa,xb,xc,xd,xe,xf],dim=1), torch.stack([za,zb,zc,zd,ze,zf],dim=1)

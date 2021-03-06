import torch
from torch import nn
from torch.nn import functional as F


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
        self.bn1 = nn.BatchNorm2d(32)
        self.c2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.c3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.Conv2d(32)

        self.pooling_size = 4
        conv_out_dim = self._calculate_output_dim(in_channels, input_height, input_width, self.pooling_size)

        self.fc1 = DenseBlock(conv_out_dim, hidden_dim)
        self.fc2 = DenseBlock(hidden_dim, hidden_dim)

        self.fc_z_out = nn.Linear(hidden_dim, latent_dim)

        self.c3_only = False

    def _calculate_output_dim(self, in_channels, input_height, input_width, pooling_size):
        x = torch.rand(1, in_channels, input_height, input_width)
        x = self.c3(self.c2(self.c1(x)))
        x = x.view(-1).unsqueeze(0).unsqueeze(0)
        x = F.max_pool1d(x, kernel_size=pooling_size)
        return x.size(-1)

    def forward(self, x):
        x = F.relu(self.bn1(self.c1(x)))
        x = F.relu(self.bn2(self.c2(x)))
        x = F.relu(self.bn3(self.c3(x)))
        if self.c3_only:
            return x
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
        self.bn1 = nn.BatchNorm2d(32)
        self.dc2 = nn.ConvTranspose2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.dc3 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

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
        x = F.relu(self.bn1(self.dc1(x)))
        x = F.relu(self.bn2(self.dc2(x)))
        x = F.relu(self.bn3(self.dc3(x)))
        x = self.dc4(x)   ### NOTE: removed the F.sigmoid on this layer - color images
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
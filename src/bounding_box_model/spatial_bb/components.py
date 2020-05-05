import torch
from torch import nn
from torch.nn import functional as F


class SpatialMappingCNN(nn.Module):
    """
    Generates spatial mapping

    match the diagram of the road
    BL FL
    B F
    BR FR
    """

    def __init__(self):
        super().__init__()
        self.f_conv = nn.Conv2d(3, 32, kernel_size=(52, 1), stride=(3, 2), padding=(1))
        self.fl_conv = nn.Conv2d(3, 32, kernel_size=(1, 50), stride=(3, 2))
        self.fr_conv = nn.Conv2d(3, 32, kernel_size=(1, 50), stride=(3, 2))

        self.b_conv = nn.Conv2d(3, 32, kernel_size=(52, 1), stride=(3, 2), padding=(1))
        self.bl_conv = nn.Conv2d(3, 32, kernel_size=(1, 50), stride=(3, 2))
        self.br_conv = nn.Conv2d(3, 32, kernel_size=(1, 50), stride=(3, 2))

        self.out_conv = nn.Conv2d(32, 32, kernel_size=(3, 3))

    def forward(self, x):
        # (b, 6, 3, 256, 306) -> (b, 32, 256, 256)

        # ---------------
        # DO NOT ROTATE THESE
        # ---------------
        bl = x[:, 3, ...]
        bl = F.relu(self.bl_conv(bl))

        fl = x[:, 0, ...]
        fl = F.relu(self.fl_conv(fl))

        # ---------------
        # ROTATE
        # ---------------
        b = x[:, 4, ...]
        #import pdb; pdb.set_trace()
        b = torch.rot90(b.half(), 1, [2, 3])
        #b = b.type_as(x[0])
        b = F.relu(self.b_conv(b))

        f = x[:, 1, ...]
        f = torch.rot90(f.half(), 1, [3, 2])
        #f = f.type_as(x[0])
        f = F.relu(self.f_conv(f))

        # ---------------
        # HORIZONTAL FLIP
        # ---------------
        br = x[:, 5, ...]
        br = torch.flip(br, [2, 3])
        #br = br.type_as(x[0])
        br = F.relu(self.br_conv(br))

        fr = x[:, 2, ...]
        fr = torch.flip(fr, [2, 3])
        #fr = fr.type_as(x[0])
        fr = F.relu(self.fr_conv(fr))

        # ---------------
        # MERGE INTO GIANT SQUARE
        # ---------------
        top = torch.cat([bl, fl], dim=3)
        mid = torch.cat([b, f], dim=3)
        bottom = torch.cat([br, fr], dim=3)
        x = torch.cat([top, mid, bottom], dim=2)

        # (b, 32, 258, 258) -> (b, 32, 256, 256)
        x = F.relu(self.out_conv(x))
        return x


class BoxesMergingCNN(nn.Module):
    """
    Merges ssl representations + spatial mapping
    """

    def __init__(self):
        super().__init__()
        self.ss_conv = nn.Conv2d(32, 32, kernel_size=(1, 24), stride=(1, 7))
        self.ss_deconv = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)

        self.up_conv_1 = nn.ConvTranspose2d(64, 32, kernel_size=8, stride=1, dilation=8)
        self.up_conv_2 = nn.ConvTranspose2d(32, 16, kernel_size=8, stride=1, dilation=8)
        self.up_conv_3 = nn.ConvTranspose2d(16, 8, kernel_size=6, stride=1, dilation=6, output_padding=2)
        self.up_conv_4 = nn.ConvTranspose2d(8, 1, kernel_size=2, stride=2)

    def forward(self, ssr, spatial_map):
        # ssr = (b, 32, 128, 918)
        # spatial_block = (b, 32, 256, 256)

        # -----------------------------
        # make all representations the same size
        # -----------------------------
        # (b, 32, 128, 128) -> (b, 32, 256, 256)
        ssr = F.relu(self.ss_conv(ssr))
        ssr = F.relu(self.ss_deconv(ssr))

        # -----------------------------
        # merge representations
        # -----------------------------
        x = torch.cat([ssr, spatial_map], dim=1)

        # -----------------------------
        # upsample back to 800 x 800
        # -----------------------------
        x = F.relu(self.up_conv_1(x))
        x = F.relu(self.up_conv_2(x))
        x = F.relu(self.up_conv_3(x))
        x = F.sigmoid(self.up_conv_4(x))

        return x




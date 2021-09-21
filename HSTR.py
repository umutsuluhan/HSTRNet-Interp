import torch
import torch.nn as nn
import numpy as np
from model.IFNet import IFNet
from model.warplayer import warp
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DownSampling(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):

        super(DownSampling, self).__init__()

        self.leaky_relu = nn.LeakyReLU(0.1)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size,
                               stride=1, padding=int((kernel_size - 1) / 2))
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size,
                               stride=1, padding=int((kernel_size - 1) / 2))

    def forward(self, x):

        x = F.avg_pool2d(x, 2)
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)

        return x


class UpSampling(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(UpSampling, self).__init__()

        self.leaky_relu = nn.LeakyReLU(0.1)
        #self.bilinear_up_sample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.pad = nn.ReplicationPad2d((0, 0, 1, 0))
        
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(
            2*out_channels, out_channels, 3, stride=1, padding=1)

    def forward(self, x, skip_con):

        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = self.conv1(x)
        x = self.leaky_relu(x)
        
        if(x.shape[2] != skip_con.shape[2]):
            x = self.pad(x)
        
        x = self.conv2(torch.cat((x, skip_con), 1))
        x = self.leaky_relu(x)

        return x


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(UNet, self).__init__()

        self.leaky_relu = nn.LeakyReLU(0.1)

        self.conv1 = nn.Conv2d(in_channels, 32, 7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(32, 32, 7, stride=1, padding=3)

        self.down1 = DownSampling(32, 64, 5)
        self.down2 = DownSampling(64, 128, 3)
        self.down3 = DownSampling(128, 256, 3)
        self.down4 = DownSampling(256, 512, 3)
        self.down5 = DownSampling(512, 512, 3)

        self.up1 = UpSampling(512, 512)
        self.up2 = UpSampling(512, 256)
        self.up3 = UpSampling(256, 128)
        self.up4 = UpSampling(128, 64)
        self.up5 = UpSampling(64, 32)

        self.conv3 = nn.Conv2d(32, out_channels, 3, stride=1, padding=1)

    def forward(self, x):

        # First hierarchy of encoder
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        s1 = self.leaky_relu(x)
      

        # Second hierarchy of encoder
        s2 = self.down1(s1)
        # Third hierarchy of encoder
        s3 = self.down2(s2)
        # Fourth hierarchy of encoder
        s4 = self.down3(s3)
        # Fifth hierarchy of encoder
        s5 = self.down4(s4)
        # Sixth hiearchy of encoder
        x = self.down5(s5)

        # First hierarchy of decoder
        x = self.up1(x, s5)
        # Second hierarchy of decoder
        x = self.up2(x, s4)
        # Third hierarchy of decoder
        x = self.up3(x, s3)
        # Fourth hierarchy of decoder
        x = self.up4(x, s2)
        # Fifth hierarchy of decoder
        x = self.up5(x, s1)

        x = self.conv3(x)
        x = self.leaky_relu(x)
        return x


class Model():

    def __init__(self):
        self.flownet = IFNet()
        self.unet = UNet(4, 32)

        self.flownet.to(device)
        self.unet.to(device)

    def inference(self, imgs, timestamps, training=False):

        # lfr_img0 = lr(t-1)
        # lfr_img1 = lr(t)
        # lfr_img2 = lr(t+1)
        # hfr_img0 = hr(t-1)
        # hfr_img1 = hr(t+1)

        lfr_img0 = imgs[:, :, :3]
        lfr_img1 = imgs[:, :, 3:6]
        hfr_img0 = imgs[:, :, 6:9]
        hfr_img1 = imgs[:, :, 9:12]
        hfr_img2 = imgs[:, :, 12:15]

        # Moving images to torch tensors
        lfr_img0 = torch.from_numpy(np.transpose(lfr_img0, (2, 0, 1))).to(
            device, non_blocking=True).unsqueeze(0).float() / 255.
        lfr_img1 = torch.from_numpy(np.transpose(lfr_img1, (2, 0, 1))).to(
            device, non_blocking=True).unsqueeze(0).float() / 255.
        hfr_img0 = torch.from_numpy(np.transpose(hfr_img0, (2, 0, 1))).to(
            device, non_blocking=True).unsqueeze(0).float() / 255.
        hfr_img1 = torch.from_numpy(np.transpose(hfr_img1, (2, 0, 1))).to(
            device, non_blocking=True).unsqueeze(0).float() / 255.
        hfr_img2 = torch.from_numpy(np.transpose(hfr_img2, (2, 0, 1))).to(
            device, non_blocking=True).unsqueeze(0).float() / 255.

        # First bi-directional optical frames extracted for intermediate flow estimation

        # lfr_prev_to_next_flow -> flow from t=0 to t=1 (high sr, low fps video)
        # lfr_next_to_prev_flow -> flow from t=1 to t=0 (high sr, low fps video)

        lfr_prev_to_next_flow,  _ = self.flownet(
            torch.cat((lfr_img0, lfr_img1), 1))
        lfr_next_to_prev_flow,  _ = self.flownet(
            torch.cat((lfr_img1, lfr_img0), 1))

        x = self.unet(lfr_prev_to_next_flow)
       
        # Maybe not necessary but think on it later
        # hfr_prev_to_t_flow, _ = self.flownet(torch.cat((hfr_img0, hfr_img1),1))
        # hfr_t_to_next_flow, _ = self.flownet(torch.cat((hfr_img1, hfr_img2),1))

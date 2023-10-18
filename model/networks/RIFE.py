import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
import torch.optim as optim
import itertools
from .warplayer import warp
from torch.nn.parallel import DistributedDataParallel as DDP
from .IFNet import *
import torch.nn.functional as F


from .Deformable_Conv.DeformableBlock import DeformableConvBlock

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes,
                                 kernel_size=4, stride=2, padding=1, bias=True),
        nn.PReLU(out_planes)
    )

class Conv2(nn.Module):
    def __init__(self, in_planes, out_planes, stride=2):
        super(Conv2, self).__init__()
        self.conv1 = conv(in_planes, out_planes, 3, stride, 1)
        self.conv2 = conv(out_planes, out_planes, 3, 1, 1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

c = 16

class ContextNet(nn.Module):
    def __init__(self):
        super(ContextNet, self).__init__()
        self.conv1 = Conv2(3, c)
        self.conv2 = Conv2(c, 2*c)
        self.conv3 = Conv2(2*c, 4*c)
        self.conv4 = Conv2(4*c, 8*c)
    
        self.deformconv1 = DeformableConvBlock(c , "add")
        self.deformconv2 = DeformableConvBlock(2*c , "add")
        self.deformconv3 = DeformableConvBlock(4*c , "add")
        self.deformconv4 = DeformableConvBlock(8*c , "add")
        self.move_to_device()

    def move_to_device(self):
        self.deformconv1.to(device)
        self.deformconv2.to(device)
        self.deformconv3.to(device)
        self.deformconv4.to(device)
        
    def deform_func(self, x, flow):
        x = self.conv1(x)
        f1 = self.deformconv1(x, flow)

        x = self.conv2(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        f2 = self.deformconv2(x, flow)

        x = self.conv3(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        f3 = self.deformconv3(x, flow)

        x = self.conv4(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        f4 = self.deformconv4(x, flow)
        return f1, f2, f3, f4

    def forward(self, x, flow):
        
        f1, f2, f3, f4 = self.deform_func(x, flow)

        f1 = F.interpolate(f1, scale_factor=2.0, mode="bilinear", align_corners=False) * 2.0
        f2 = F.interpolate(f2, scale_factor=2.0, mode="bilinear", align_corners=False) * 2.0
        f3 = F.interpolate(f3, scale_factor=2.0, mode="bilinear", align_corners=False) * 2.0
        f4 = F.interpolate(f4, scale_factor=2.0, mode="bilinear", align_corners=False) * 2.0
        return [f1, f2, f3, f4]

class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()
        self.down0 = Conv2(15, 2*c, 1)
        self.down1 = Conv2(6*c, 6*c)
        self.down2 = Conv2(14*c, 14*c)
        self.down3 = Conv2(30*c, 30*c)
        self.up0 = deconv(62*c, 8*c)
        self.up1 = deconv(22*c, 4*c)
        self.up2 = deconv(10*c, 2*c)
        self.up3 = deconv(4*c, c)
        self.conv = nn.Conv2d(c, 4, 3, 2, 1)

    def forward(self, warped_lr_img1_0, lr_img1, warped_lr_img1_2, warped_hr_img1_0, warped_hr_img1_2, c0, c1, c2, c3):
        s0 = self.down0(torch.cat((warped_lr_img1_0, lr_img1, warped_lr_img1_2, warped_hr_img1_0, warped_hr_img1_2), 1))
        s1 = self.down1(torch.cat((s0, c0[0], c1[0], c2[0], c3[0]), 1))
        s2 = self.down2(torch.cat((s1, c0[1], c1[1], c2[1], c3[1]), 1))
        s3 = self.down3(torch.cat((s2, c0[2], c1[2], c2[2], c3[2]), 1))
        x = self.up0(torch.cat((s3, c0[3], c1[3], c2[3], c3[3]), 1))
        x = self.up1(torch.cat((x, s2), 1))
        x = self.up2(torch.cat((x, s1), 1))
        x = self.up3(torch.cat((x, s0), 1))
        x = self.conv(x)
        return x

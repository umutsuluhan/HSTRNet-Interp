import torch
import torch.nn as nn
import cv2
import time
import numpy as np
from model.IFNet import IFNet
from model.warplayer import warp
import torch.nn.functional as F

device = 'cpu' #torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

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
        self.pad1 = nn.ReplicationPad2d((0, 0, 1, 0))
        self.pad2 = nn.ReplicationPad2d((0, 1, 0, 0))

        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(
            2*out_channels, out_channels, 3, stride=1, padding=1)

    def forward(self, x, skip_con):

        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = self.conv1(x)
        x = self.leaky_relu(x)

        if(x.shape[2] != skip_con.shape[2]):
            x = self.pad1(x)
        if(x.shape[3] != skip_con.shape[3]):
            x = self.pad2(x)
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


class backWarp(nn.Module):
    """
    A class for creating a backwarping object.
    This is used for backwarping to an image:
    Given optical flow from frame I0 to I1 --> F_0_1 and frame I1, 
    it generates I0 <-- backwarp(F_0_1, I1).
    ...
    Methods
    -------
    forward(x)
        Returns output tensor after passing input `img` and `flow` to the backwarping
        block.
    """

    def __init__(self, W, H, device):
        """
        Parameters
        ----------
            W : int
                width of the image.
            H : int
                height of the image.
            device : device
                computation device (cpu/cuda). 
        """

        super(backWarp, self).__init__()
        # create a grid
        gridX, gridY = np.meshgrid(np.arange(W), np.arange(H))
        self.W = W
        self.H = H
        self.gridX = torch.tensor(gridX, requires_grad=False, device=device)
        self.gridY = torch.tensor(gridY, requires_grad=False, device=device)

    def forward(self, img, nd_flow):
        """
        Returns output tensor after passing input `img` and `flow` to the backwarping
        block.
        I0  = backwarp(I1, F_0_1)
        Parameters
        ----------
            img : tensor
                frame I1.
            flow : tensor
                optical flow from I0 and I1: F_0_1.
        Returns
        -------
            tensor
                frame I0.
        """

        if nd_flow.dtype == 'float32':    
            flow = torch.from_numpy(nd_flow)
        else:
            flow = nd_flow
        
        flow = flow.to(device)
        if (flow.shape[2] != img.shape[2]):
            flow = F.interpolate(flow, scale_factor=2, mode="bilinear",    # ASK IF RIGHT THING TO DO
                             align_corners=False)

        # Extract horizontal and vertical flows.
        u = flow[:, 0, :, :]
        v = flow[:, 1, :, :]
        x = self.gridX.unsqueeze(0).expand_as(u).float() + u
        y = self.gridY.unsqueeze(0).expand_as(v).float() + v
        # range -1 to 1
        x = 2*(x/self.W - 0.5)
        y = 2*(y/self.H - 0.5)
        # stacking X and Y
        grid = torch.stack((x, y), dim=3)
        # Sample pixels using bilinear interpolation.
        imgOut = torch.nn.functional.grid_sample(img, grid)
        imgOut = imgOut.to(device)
        return imgOut


class Model():

    def __init__(self):
        self.flownet = IFNet()
        self.unet = UNet(4, 32)
        self.arbitrary_time_f_int = UNet(29, 5)

        self.arbitrary_time_f_int.to(device)
        self.flownet.to(device)
        self.unet.to(device)

    def optical_flow_est(self, x):
        x = F.interpolate(x, scale_factor=0.5, mode="bilinear",
                          align_corners=False)
        img0 = x[:, :3].cpu().numpy()
        img1 = x[:, 3:].cpu().numpy()

        num_samples, _, x, y = img0.shape
        flow_batch = np.empty((0, 2, x, y))
        flow_time = []
        for i in range(num_samples):
            img0_single = img0[i, :, :, :].reshape(x, y, 3)
            img1_single = img1[i, :, :, :].reshape(x, y, 3)
            img0_single = cv2.cvtColor(img0_single, cv2.COLOR_BGR2GRAY)
            img1_single = cv2.cvtColor(img1_single, cv2.COLOR_BGR2GRAY)

            start2 = time.time()
            flow_single = cv2.calcOpticalFlowFarneback(img0_single, img1_single, None, pyr_scale=0.2, levels=3,
                                                       winsize=15, iterations=1, poly_n=1, poly_sigma=1.2, flags=0)
            end2 = time.time()
            flow_time.append((end2 - start2) * 1000)
            flow_single = flow_single.reshape(1, 2, x, y)
            flow_batch = np.append(flow_batch, flow_single, axis=0)
        return torch.tensor(flow_batch, dtype=torch.float, device=device)

    def intermediate_flow_est(self, x, t):

        F_0_1 = x[:, :2].cpu().numpy()
        F_1_0 = x[:, :2].cpu().numpy()

        temp = -t * (1 - t)
        fCoeff = [temp, t * t, (1 - t) * (1 - t), temp]

        F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
        F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0

        return F_t_0, F_t_1

    def inference(self, imgs, timestamps, training=False):

        t = 0.5        

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

        lfr_F_0_1 = self.optical_flow_est(               # Flow from t=0 to t=1 (high sr, low fps video)
            torch.cat((lfr_img0, lfr_img1), 1))
        lfr_F_1_0 = self.optical_flow_est(               # Flow from t=1 to t=0 (high sr, low fps video)
            torch.cat((lfr_img1, lfr_img0), 1))

        hfr_F_0_1 = self.optical_flow_est(               # Flow from t=0 to t=1 (low sr, high fps video)
            torch.cat((hfr_img0, hfr_img1), 1))
        hfr_F_1_0 = self.optical_flow_est(               # Flow from t=1 to t=0 (low sr, high fps video)
            torch.cat((hfr_img1, hfr_img0), 1))

        # hfr_F_1_2 = self.optical_flow_est(               # Flow from t=1 to t=2 (low sr, high fps video)
        #     torch.cat((hfr_img1, hfr_img2), 1))

        hfr_F_2_1 = self.optical_flow_est(               # Flow from t=2 to t=1 (low sr, high fps video)
            torch.cat((hfr_img2, hfr_img1), 1))

        F_t_0, F_t_1 = self.intermediate_flow_est(       # Flow from t to 0 and flow from t to 1 using provided low fps video frames
            torch.cat((lfr_F_0_1, lfr_F_1_0), 1), 0.5)  

        F_t_0 = torch.from_numpy(F_t_0).to(device)
        F_t_1 = torch.from_numpy(F_t_1).to(device)

        backwarp = backWarp(lfr_img0.shape[3], lfr_img0.shape[2], device)   # Backwarping module
        backwarp.to(device)

        #I0  = backwarp(I1, F_0_1)

        g_I0_F_t_0 = backwarp(lfr_img0, F_t_0)              # Backwarp of I0 and F_t_0
        g_I1_F_t_1 = backwarp(lfr_img1, F_t_1)              # Backwarp of I1 and F_t_1
        
        warped_lfr_img0 = backwarp(lfr_img1, lfr_F_0_1)     # Backwarp of LFR_I0 and F_t_0
        warped_lfr_img1 = backwarp(lfr_img0, lfr_F_1_0)     # Backwarp of LFR_I1 and F_t_0
        
        warped_hfr_img0 = backwarp(hfr_img1, hfr_F_0_1)     # Backwarp of HFR_I0 and F_t_0
        warped_hfr_img1 = backwarp(hfr_img0, hfr_F_1_0)     # Backwarp of HFR_I1 and F_t_0
        warped_hfr_img2 = backwarp(hfr_img1, hfr_F_2_1)     # Backwarp of HFR_I2 and F_t_0

        # Interpolation of flows to match tensor sizes

        lfr_F_0_1 = F.interpolate(lfr_F_0_1, scale_factor=2, mode="bilinear",    
                             align_corners=False) 
        lfr_F_1_0 = F.interpolate(lfr_F_1_0, scale_factor=2, mode="bilinear",   
                             align_corners=False) 
        F_t_1 = F.interpolate(F_t_1, scale_factor=2, mode="bilinear", 
                             align_corners=False)
        F_t_0 = F.interpolate(F_t_0, scale_factor=2, mode="bilinear",  
                             align_corners=False)


        intrpOut = self.arbitrary_time_f_int(torch.cat((warped_lfr_img0, warped_lfr_img1, warped_hfr_img0, warped_hfr_img1, 
                                                        warped_hfr_img2, lfr_F_0_1, lfr_F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0), 
                                                       dim=1))

        F_t_0_f = intrpOut[:, :2, :, :] + F_t_0
        F_t_1_f = intrpOut[:, 2:4, :, :] + F_t_1
        V_t_0   = torch.sigmoid(intrpOut[:, 4:5, :, :])
        V_t_1   = 1 - V_t_0

        g_I0_F_t_0_f = backwarp(lfr_img0, F_t_0_f)
        g_I1_F_t_1_f = backwarp(lfr_img1, F_t_1_f)

        wCoeff = [1 - t, t]

        Ft_p = (wCoeff[0] * V_t_0 * g_I0_F_t_0_f + wCoeff[1] * V_t_1 * g_I1_F_t_1_f) / (wCoeff[0] * V_t_0 + wCoeff[1] * V_t_1)
        
        result = Ft_p.detach().numpy()
        result = result[0,:]
        result = np.transpose(result, (1, 2, 0))
        cv2.imshow("result", result)
        cv2.waitKey(0)

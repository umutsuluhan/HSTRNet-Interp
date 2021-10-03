# Model implementation of Farneback + Super_Slomo

import torch
import torch.nn as nn
import cv2
import time
import math
import numpy as np
import sys
import matplotlib.pyplot as plt
import torch.nn.functional as F
from .unet_model import UNet
from .backwarp import backWarp

device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HSTR_FSS():

    def __init__(self):
        self.unet = UNet(15, 3)
        self.unet.to(device)

    # def flow2rgb(self, flow_map_np):
    #     h, w, _ = flow_map_np.shape
    #     rgb_map = np.ones((h, w, 3)).astype(np.float32)
    #     normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
    
    #     rgb_map[:, :, 0] += normalized_flow_map[:, :, 0]
    #     rgb_map[:, :, 1] -= 0.5 * (normalized_flow_map[:, :, 0] + normalized_flow_map[:, :, 1])
    #     rgb_map[:, :, 2] += normalized_flow_map[:, :, 1]
    #     return rgb_map.clip(0, 1)

    def return_parameters(self):
        return list(self.unet.parameters())

    def optical_flow_est(self, x):

        # Optical flow method which employs Farneback method to extract flow of each pixel in the image (dense optical flow).

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
            
            
            # Flow debug kodu
            # image = self.flow2rgb(flow_single)
            # cv2.imshow("win", image)
            # cv2.waitKey(1000)
            
            end2 = time.time()
            flow_time.append((end2 - start2) * 1000)
            flow_single = flow_single.reshape(1, 2, x, y)
            flow_batch = np.append(flow_batch, flow_single, axis=0)
        return torch.tensor(flow_batch, dtype=torch.float, device=device)

    def intermediate_flow_est(self, x, t):

        F_0_1 = x[:, :2].cpu().numpy()
        F_1_0 = x[:, 2:4].cpu().numpy()

        temp = -t * (1 - t)
        fCoeff = [temp, t * t, (1 - t) * (1 - t), temp]

        F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
        F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0

        return F_t_0, F_t_1

    def inference(self, imgs, timestamps, training=False):

        t = 0.5                        # Timestamp of the generated frame.

        if(training == False):
            hr_img0 = imgs[:, :, :3]      # hr_img0 = hr(t-1)
            hr_img1 = imgs[:, :, 3:6]     # hr_img1 = hr(t+1)
            lr_img0 = imgs[:, :, 6:9]     # lr_img0 = lr(t-1)
            lr_img1 = imgs[:, :, 9:12]    # lr_img1 = lr(t)
            lr_img2 = imgs[:, :, 12:15]   # lr_img2 = lr(t+1)
            
    
            # Moving images to torch tensors
            hr_img0 = torch.from_numpy(np.transpose(hr_img0, (2, 0, 1))).to(
                device, non_blocking=True).unsqueeze(0).float() / 255.
            hr_img1 = torch.from_numpy(np.transpose(hr_img1, (2, 0, 1))).to(
                device, non_blocking=True).unsqueeze(0).float() / 255.
            lr_img0 = torch.from_numpy(np.transpose(lr_img0, (2, 0, 1))).to(
                device, non_blocking=True).unsqueeze(0).float() / 255.
            lr_img1 = torch.from_numpy(np.transpose(lr_img1, (2, 0, 1))).to(
                device, non_blocking=True).unsqueeze(0).float() / 255.
            lr_img2 = torch.from_numpy(np.transpose(lr_img2, (2, 0, 1))).to(
                device, non_blocking=True).unsqueeze(0).float() / 255.
            
        else:
            hr_img0 = imgs[:, :3]      # hr_img0 = hr(t-1)
            hr_img1 = imgs[:, 3:6]     # hr_img1 = hr(t+1)
            lr_img0 = imgs[:, 6:9]     # lr_img0 = lr(t-1)
            lr_img1 = imgs[:, 9:12]    # lr_img1 = lr(t)
            lr_img2 = imgs[:, 12:15]   # lr_img2 = lr(t+1)    
        
        # First bi-directional optical frames extracted for intermediate flow estimation

        hr_F_0_1 = self.optical_flow_est(               # Flow from t=0 to t=1 (high sr, low fps video)
            torch.cat((hr_img0, hr_img1), 1))

        hr_F_1_0 = self.optical_flow_est(               # Flow from t=1 to t=0 (high sr, low fps video)
            torch.cat((hr_img1, hr_img0), 1))

        lr_F_0_1 = self.optical_flow_est(               # Flow from t=0 to t=1 (low sr, high fps video)
            torch.cat((lr_img0, lr_img1), 1))
       
        lr_F_2_1 = self.optical_flow_est(               # Flow from t=2 to t=1 (low sr, high fps video)
            torch.cat((lr_img2, lr_img1), 1))

        F_t_0, F_t_1 = self.intermediate_flow_est(       # Flow from t to 0 and flow from t to 1 using provided low fps video frames
            torch.cat((hr_F_0_1, hr_F_1_0), 1), 0.5)

        F_t_0 = torch.from_numpy(F_t_0).to(device)
        F_t_1 = torch.from_numpy(F_t_1).to(device)

        # Backwarping module
        backwarp = backWarp(hr_img0.shape[3], hr_img0.shape[2], device)
        backwarp.to(device)

        #I0  = backwarp(I1, F_0_1)

        # Backwarp of I0 and F_t_0
        g_I0_F_t_0 = backwarp(hr_img0, F_t_0)
        # Backwarp of I1 and F_t_1
        g_I1_F_t_1 = backwarp(hr_img1, F_t_1)

        # Backwarp of LR_I0 and F_t_0
        warped_lr_img0 = backwarp(lr_img1, lr_F_0_1)

        # Backwarp of LR_I2 and F_t_0
        warped_lr_img2 = backwarp(lr_img1, lr_F_2_1)

        # Interpolation of flows to match tensor sizes

        input_imgs = torch.cat(
            (warped_lr_img0,  lr_img1, warped_lr_img2, g_I0_F_t_0, g_I1_F_t_1), dim=1)

        padding1_mult = math.floor(input_imgs.shape[2] / 32) + 1
        padding2_mult = math.floor(input_imgs.shape[3] / 32) + 1
        pad1 = (32 * padding1_mult) - input_imgs.shape[2]
        pad2 = (32 * padding2_mult) - input_imgs.shape[3]

        # Padding to meet dimension requirements of the network
        # Done before network call, otherwise slows down the network
        padding1 = nn.ReplicationPad2d((0, pad2, pad1, 0))

        input_imgs = padding1(input_imgs)

        Ft_p = self.unet(input_imgs)

        padding2 = nn.ReplicationPad2d((0, -pad2, -pad1, 0))
        Ft_p = padding2(Ft_p)

        result = Ft_p.detach().numpy()
        result = result[0, :]
        result = np.transpose(result, (1, 2, 0))
        # cv2.imshow("win", result)
        # cv2.waitKey(100)

        return result

# Model implementation of Farneback + Super_Slomo

import torch
import torch.nn as nn
import cv2
import time
import math
import numpy as np
import sys
import argparse
import matplotlib.pyplot as plt
import torch.nn.functional as F
from collections import OrderedDict
from unet_model import UNet
from backwarp import backWarp
from RAFT.core.raft import RAFT
#from pytorch_msssim import ssim

class HSTR_RAFT():
    def __init__(self, device, args):
        self.device = device
        self.unet = UNet(15, 3)
        self.flownet = RAFT(args)
        
        self.flownet.to(device)
        self.unet.to(self.device)


    def return_parameters(self):
        return list(self.unet.parameters())

    def intermediate_flow_est(self, x, t):

        F_0_1 = x[:, :2].detach().cpu().numpy()
        F_1_0 = x[:, 2:4].detach().cpu().numpy()

        temp = -t * (1 - t)
        fCoeff = [temp, t * t, (1 - t) * (1 - t), temp]

        F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
        F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0

        return F_t_0, F_t_1

    def inference(self, imgs, training=False):
        
        hr_img0 = imgs[:, :3]      # hr_img0 = hr(t-1)
        hr_img1 = imgs[:, 3:6]     # hr_img1 = hr(t+1)
        lr_img0 = imgs[:, 6:9]     # lr_img0 = lr(t-1)
        lr_img1 = imgs[:, 9:12]    # lr_img1 = lr(t)
        lr_img2 = imgs[:, 12:15]   # lr_img2 = lr(t+1)    

        state_dict = torch.load("/home/hus/Desktop/repos/HSTRNet/HSTR_RAFT/trained_models/raft-small.pth")
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v

        self.flownet.load_state_dict(new_state_dict)
        self.flownet.eval()

        hr_F_0_1 = self.flownet(hr_img0, hr_img1)       # Flow from t=0 to t=1 (high sr, low fps video)
        print(hr_F_0_1[1].shape)
        
        hr_F_0_1 = hr_F_0_1 * 2

        hr_F_1_0 = self.flownet(hr_img1, hr_img0)       # Flow from t=1 to t=0 (high sr, low fps video)
            
        
        hr_F_1_0 = hr_F_1_0 * 2

        lr_F_1_0 = self.flownet(lr_img1, lr_img0)       # Flow from t=0 to t=1 (low sr, high fps video)
           
        
        lr_F_1_0 = lr_F_1_0 * 2
       
        lr_F_1_2 = self.flownet(lr_img1, lr_img2)       # Flow from t=2 to t=1 (low sr, high fps video)
            
        
        lr_F_1_2 = lr_F_1_2 * 2

        F_t_0, F_t_1 = self.intermediate_flow_est(       # Flow from t to 0 and flow from t to 1 using provided low fps video frames
            torch.cat((hr_F_0_1[1], hr_F_1_0[1]), 1), 0.5)

        F_t_0 = torch.from_numpy(F_t_0).to(self.device)
        F_t_1 = torch.from_numpy(F_t_1).to(self.device)

        # Backwarping module
        backwarp = backWarp(hr_img0.shape[3], hr_img0.shape[2], self.device)

        #I0  = backwarp(I1, F_0_1)

        # Backwarp of I0 and F_t_0
        g_I0_F_t_0 = backwarp(hr_img0, F_t_0)
        # Backwarp of I1 and F_t_1
        g_I1_F_t_1 = backwarp(hr_img1, F_t_1)
        
        MSE_loss_fn = nn.MSELoss()
        
        MSE_loss = MSE_loss_fn(g_I0_F_t_0, hr_img0)
        
        # result = g_I0_F_t_0.cpu().detach().numpy()
        # result = result[0, :]
        # result = np.transpose(result, (1, 2, 0))
        # cv2.imshow("win", result)
        # cv2.waitKey(10000)
        
        # result = hr_img0.cpu().detach().numpy()
        # result = result[0, :]
        # result = np.transpose(result, (1, 2, 0))
        # cv2.imshow("win1", result)
        # cv2.waitKey(10000)
        
        psnr1 = float((10 * math.log10(1 / MSE_loss.item())))
 #       ssim1 = float(ssim(g_I0_F_t_0, hr_img0))
        
        MSE_loss = MSE_loss_fn(g_I1_F_t_1, hr_img1)
        psnr2 = float((10 * math.log10(1 / MSE_loss.item())))
 #       ssim2 = float(ssim(g_I1_F_t_1, hr_img1))

        print(psnr1)
 #       print(ssim1)
        print(psnr2)
 #       print(ssim2)

        warped_lr_img1_0 = backwarp(lr_img0, lr_F_1_0[1])
        warped_lr_img1_2 = backwarp(lr_img2, lr_F_1_2[1])


        input_imgs = torch.cat(
            (warped_lr_img1_0,  lr_img1, warped_lr_img1_2, g_I0_F_t_0, g_I1_F_t_1), dim=1)


        Ft_p = self.unet(input_imgs)

        result = Ft_p

        if training == False:
            return result
        else:
            return result, g_I0_F_t_0, g_I1_F_t_1, warped_lr_img1_0, lr_img0, warped_lr_img1_2, lr_img2
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', default = "True", help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()
    
    
    img0_HR = cv2.imread(
        "/home/hus/Desktop/data/vimeo_triplet/sequences/00001/0001/im1.png")
    img1_HR = cv2.imread(
        "/home/hus/Desktop/data/vimeo_triplet/sequences/00001/0001/im3.png")

    img0_LR = cv2.imread(
        "/home/hus/Desktop/data/vimeo_triplet/sequences/00001/0001/im1.png")
    img1_LR = cv2.imread(
        "/home/hus/Desktop/data/vimeo_triplet/sequences/00001/0001/im2.png")
    img2_LR = cv2.imread(
        "/home/hus/Desktop/data/vimeo_triplet/sequences/00001/0001/im3.png")
                                                        
    device = "cpu"

    img0_HR = torch.from_numpy(np.transpose(img0_HR, (2, 0, 1))).to(
        device, non_blocking=True).unsqueeze(0).float() / 255.
    img1_HR = torch.from_numpy(np.transpose(img1_HR, (2, 0, 1))).to(
        device, non_blocking=True).unsqueeze(0).float() / 255.
    img0_LR = torch.from_numpy(np.transpose(img0_LR, (2, 0, 1))).to(
        device, non_blocking=True).unsqueeze(0).float() / 255.
    img1_LR = torch.from_numpy(np.transpose(img1_LR, (2, 0, 1))).to(
        device, non_blocking=True).unsqueeze(0).float() / 255.
    img2_LR = torch.from_numpy(np.transpose(img2_LR, (2, 0, 1))).to(
        device, non_blocking=True).unsqueeze(0).float() / 255.

    imgs = torch.cat((img0_HR, img1_HR, img0_LR, img1_LR, img2_LR), 1)
    model = HSTR_RAFT(device, args)

    result = model.inference(imgs).cpu().detach().numpy()
    result = result[0, :]
    result = np.transpose(result, (1, 2, 0))
    cv2.imshow("win", result)
    cv2.waitKey(10000)

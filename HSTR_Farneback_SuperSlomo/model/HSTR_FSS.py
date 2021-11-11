# Model implementation of Farneback + Super_Slomo

import torch
import torch.nn as nn
import cv2
import time
import math
import numpy as np
import torch.nn.functional as F
from .unet_model import UNet
from .warplayer import warp
from .IFNet import IFNet

class HSTR_FSS():
    def __init__(self, device):
        self.device = device
        
        self.flownet = IFNet()
        self.flownet.to(device)
        
        self.unet = UNet(15, 3)
        self.unet.to(self.device)

    def return_parameters(self):
        return list(self.unet.parameters())

    def optical_flow_est(self, x):

        # Optical flow method which employs Farneback method to extract flow of each pixel in the image (dense optical flow).

        img0 = (x[:, :3].cpu().numpy()*255).astype('uint8')
        img1 = (x[:, 3:].cpu().numpy()*255).astype('uint8')

        num_samples, _, x, y = img0.shape
        flow_batch = np.empty((0, 2, x, y))
        flow_time = []
        for i in range(num_samples):
            img0_single = np.transpose(img0[i, :, :, :], (1, 2, 0))
            img1_single = np.transpose(img1[i, :, : ,:], (1, 2, 0))
            img0_single = cv2.cvtColor(img0_single, cv2.COLOR_BGR2GRAY)
            img1_single = cv2.cvtColor(img1_single, cv2.COLOR_BGR2GRAY)

            start2 = time.time()
            flow_single = cv2.calcOpticalFlowFarneback(img0_single, img1_single, None, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

            end2 = time.time()
            flow_time.append((end2 - start2) * 1000)
            flow_single = np.transpose(flow_single, (2, 0, 1))
            flow_single = flow_single[np.newaxis, :]
            
            flow_batch = np.append(flow_batch, flow_single, axis=0)
        return torch.tensor(flow_batch, dtype=torch.float, device=self.device)

    def intermediate_flow_est(self, x, t):

        F_0_1 = x[:, :2].cpu().numpy()
        F_1_0 = x[:, 2:4].cpu().numpy()

        temp = -t * (1 - t)
        fCoeff = [temp, t * t, (1 - t) * (1 - t), temp]

        F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
        F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0

        return F_t_0, F_t_1
    
    def convert(self, param):
        return {
        k.replace("module.", ""): v
            for k, v in param.items()
            if "module." in k
        }

    def inference(self, imgs, timestamps, training=False):
        
        hr_img0 = imgs[:, :3]      # hr_img0 = hr(t-1)
        hr_img1 = imgs[:, 3:6]     # hr_img1 = hr(t+1)
        lr_img0 = imgs[:, 6:9]     # lr_img0 = lr(t-1)
        lr_img1 = imgs[:, 9:12]    # lr_img1 = lr(t)
        lr_img2 = imgs[:, 12:15]   # lr_img2 = lr(t+1)   
        
        # Flow of the low resolution images
        lr_F_1_0 = self.optical_flow_est(               
            torch.cat((lr_img1, lr_img0), 1))
        
        lr_F_1_2 = self.optical_flow_est(               
            torch.cat((lr_img1, lr_img2), 1))
        
    
        hr_images = torch.cat((hr_img0, hr_img1), 1)

        # Warped hr_images taken from IFNet
        warped_hr_img1_0, warped_hr_img1_2 = self.flownet(hr_images)


        # Warping lr_images with RIFE warp module
        warped_lr_img1_0 = warp(lr_img0, lr_F_1_0)
        warped_lr_img1_2 = warp(lr_img2, lr_F_1_2)


        input_imgs = torch.cat(
            (warped_lr_img1_0,  lr_img1, warped_lr_img1_2, warped_hr_img1_0, warped_hr_img1_2), dim=1)

        result = self.unet(input_imgs)

        if training == False:
            return result
        else:
            return result, warped_hr_img1_0, warped_hr_img1_2, warped_lr_img1_0, lr_img0, warped_lr_img1_2, lr_img2
        
if __name__ == '__main__':
    img0_HR = cv2.imread(
        "/home/hus/Desktop/data/vimeo_triplet/sequences/00001/0412/im1.png")
    img2_HR = cv2.imread(
        "/home/hus/Desktop/data/vimeo_triplet/sequences/00001/0412/im3.png")

    img0_LR = cv2.imread(
        "/home/hus/Desktop/data/vimeo_triplet_lr/sequences/00001/0412/im1.png")
    img1_LR = cv2.imread(
        "/home/hus/Desktop/data/vimeo_triplet_lr/sequences/00001/0412/im2.png")
    img2_LR = cv2.imread(
        "/home/hus/Desktop/data/vimeo_triplet_lr/sequences/00001/0412/im3.png")

    padding1_mult = math.floor(img0_HR.shape[0] / 32) + 1
    padding2_mult = math.floor(img0_HR.shape[1] / 32) + 1
    pad1 = (32 * padding1_mult) - img0_HR.shape[0]
    pad2 = (32 * padding2_mult) - img0_HR.shape[1]

    # Padding to meet dimension requirements of the network
    # Done before network call, otherwise slows down the network
    padding1 = nn.ReplicationPad2d((0, pad2, pad1, 0))

    device = "cpu"

    img0_HR = torch.from_numpy(np.transpose(img0_HR, (2, 0, 1))).to(
        device, non_blocking=True).unsqueeze(0).float() / 255.
    img2_HR = torch.from_numpy(np.transpose(img2_HR, (2, 0, 1))).to(
        device, non_blocking=True).unsqueeze(0).float() / 255.
    img0_LR = torch.from_numpy(np.transpose(img0_LR, (2, 0, 1))).to(
        device, non_blocking=True).unsqueeze(0).float() / 255.
    img1_LR = torch.from_numpy(np.transpose(img1_LR, (2, 0, 1))).to(
        device, non_blocking=True).unsqueeze(0).float() / 255.
    img2_LR = torch.from_numpy(np.transpose(img2_LR, (2, 0, 1))).to(
        device, non_blocking=True).unsqueeze(0).float() / 255.

    imgs = torch.cat((img0_HR, img2_HR, img0_LR, img1_LR, img2_LR), 1)
    # imgs = padding1(imgs)
    model = HSTR_FSS(device)
    #model.eval()

    result = model.inference(imgs, []).cpu().detach().numpy()
    result = result[0, :]
    result = np.transpose(result, (1, 2, 0))
    cv2.imshow("win", result)
    cv2.waitKey(3000)

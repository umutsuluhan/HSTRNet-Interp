import torch
import torch.nn as nn
import cv2
import time
import math
import numpy as np
import sys
import argparse
import io
import torch.nn.functional as F

from model.RIFE_v5.warplayer import warp
from model.RIFE_v5.IFNet import IFNet
from model.RIFE_v5.RIFE import ContextNet, FusionNet

class HSTRNet:
    def __init__(self, device):
        self.device = device
        
        self.ifnet = IFNet()
        self.contextnet = ContextNet()
        self.unet = FusionNet()
        
        self.ifnet.to(self.device)
        self.contextnet.to(self.device)
        self.unet.to(self.device)
        
    def return_parameters(self):
        param_list = list(self.contextnet.parameters())
        for param in self.unet.parameters():
            param_list.append(param)
        # print(param_list)
        # print(list(self.unet.parameters()))
        return param_list
    
    def convert(self, param):
        return {
        k.replace("module.", ""): v
            for k, v in param.items()
            if "module." 
        }
    
    def convert_to_numpy(self, img):
        result = img.cpu().detach().numpy()
        result = result[0, :]
        result = np.transpose(result, (1, 2, 0))
        return result * 255

    #-------------------------------------------------------------------
    def homography(self, img):
        img = self.convert_to_numpy(img)
        
        homography = np.zeros((3,3))
        homography[0][0] = 1
        homography[1][1] = 1
        homography[2][2] = 1

        #homography = homography + (0.00000000001 ** 0.5) * np.random.randn(homography.shape[0], homography.shape[1])
        homography = homography + (0.0000000005 ** 0.5) * np.random.randn(homography.shape[0], homography.shape[1])
        homography[2][2] = 1

        homography_img = cv2.warpPerspective(img, homography, (img.shape[1], img.shape[0]))

        homography_img = torch.from_numpy(np.transpose(homography_img, (2, 0, 1))).to(
            self.device, non_blocking=True).unsqueeze(0).float() / 255.

        return homography_img
    #-------------------------------------------------------------------

    def inference(self, lr_images, hr_images, gt=None):
        lr_img0 = lr_images[:, :3]
        lr_img1 = lr_images[:, 3:6]
        lr_img2 = lr_images[:, 6:9]
        
        #start_time_rife = time.time()
        warped_hr_img1_0, warped_hr_img1_2, flow = self.ifnet(hr_images)   
        # psnr = -10 * math.log10(((gt - warped_hr_img1_0) * (gt - warped_hr_img1_0)).mean())
        # print("first " + str(psnr))
        # psnr = -10 * math.log10(((gt - warped_hr_img1_2) * (gt - warped_hr_img1_2)).mean())
        # print("second " + str(psnr))
        _, _, lr_F_1_0_ = self.ifnet(torch.cat((lr_img1, lr_img0),1))
        _, _, lr_F_1_2_ = self.ifnet(torch.cat((lr_img1, lr_img2),1))
        #rife_time = time.time() - start_time_rife
        lr_F_1_0 = lr_F_1_0_ * -2.0
        lr_F_1_2 = lr_F_1_2_ * 2.0
        
        lr_F_1_0 = F.interpolate(lr_F_1_0, scale_factor=2.0, mode="bilinear",
                              align_corners=False) * 2.0
        
        lr_F_1_2 = F.interpolate(lr_F_1_2, scale_factor=2.0, mode="bilinear",
                              align_corners=False) * 2.0
        
        # REPLACE BELOW WITH DEFORMABLE CONV LATER
        #start_time_warp = time.time()
        warped_lr_img1_0 = warp(lr_img0, lr_F_1_0[:, :2], self.device)
        warped_lr_img1_2 = warp(lr_img2, lr_F_1_2[:, 2:4], self.device)
        # psnr = -10 * math.log10(((lr_img1 - warped_lr_img1_0) * (lr_img1 - warped_lr_img1_0)).mean())
        # print("first " + str(psnr))
        # x = warp(lr_img2, lr_F_1_2[:, :2], self.device)
        # psnr_ = -10 * math.log10(((lr_img1 - x) * (lr_img1 - x)).mean())
        # print("second " + str(psnr_))
        # x = warp(lr_img2, lr_F_1_2, self.device)
        # psnr_ = -10 * math.log10(((lr_img1 - x) * (lr_img1 - x)).mean())
        # print("Third " + str(psnr_))
        #warp_time = time.time() - start_time_warp
        
        # flow = F.interpolate(flow, scale_factor=2.0, mode="bilinear",
        #                     align_corners=False) * 2.0
        
        # MAYBE USE LATER WITH REAL IMAGES
        #-------------------------------------------------------------------
        #warped_hr_img1_0 = self.homography(warped_hr_img1_0)
        #warped_hr_img1_2 = self.homography(warped_hr_img1_2)
        #warped_lr_img1_0 = self.homography(warped_lr_img1_0)
        #warped_lr_img1_2 = self.homography(warped_lr_img1_2)
        #-------------------------------------------------------------------


        #start_time_context = time.time()
        c0_HR = self.contextnet(hr_images[:, :3], flow[:, :2])
        c1_HR = self.contextnet(hr_images[:, 3:6], flow[:, 2:4])
        c0_LR = self.contextnet(lr_img0, lr_F_1_0_[:, :2])
        c1_LR = self.contextnet(lr_img2, lr_F_1_2_[:, 2:4])
        #context_time = time.time() - start_time_context

        torch.cuda.empty_cache()
        #start_time_fusion = time.time()        
        refine_output = self.unet(warped_lr_img1_0, lr_img1, warped_lr_img1_2, warped_hr_img1_0, warped_hr_img1_2, c0_HR, c1_HR, c0_LR, c1_LR)
        #fusion_time = time.time() - start_time_fusion
        
        res = torch.sigmoid(refine_output[:, :3]) * 2 - 1
        mask = torch.sigmoid(refine_output[:, 3:4])
        merged_img = warped_hr_img1_0 * mask + warped_hr_img1_2 * (1 - mask)
        pred = merged_img + res
        pred = torch.clamp(pred, 0, 1)

        return pred
        #return pred, rife_time, warp_time, context_time, fusion_time
"""
if __name__ == "__main__":
    hr_img0 = cv2.imread(
        "/home/hus/Desktop/data/vimeo_triplet/sequences/00001/0412/im1.png")

    gt = cv2.imread(
        "/home/hus/Desktop/data/vimeo_triplet/sequences/00001/0412/im2.png"
    )

    hr_img1 = cv2.imread(
        "/home/hus/Desktop/data/vimeo_triplet/sequences/00001/0412/im3.png"
    )

    lr_img0 = cv2.imread(
        "/home/hus/Desktop/data/vimeo_triplet_lr/sequences/00001/0412/im1.png"
    )
    lr_img1 = cv2.imread(
        "/home/hus/Desktop/data/vimeo_triplet_lr/sequences/00001/0412/im2.png"
    )
    lr_img2 = cv2.imread(
        "/home/hus/Desktop/data/vimeo_triplet_lr/sequences/00001/0412/im3.png"
    )
    
    device = "cpu"

    hr_img0 = torch.from_numpy(np.transpose(hr_img0, (2, 0, 1))).to(
        device, non_blocking=True).unsqueeze(0).float() / 255.
    hr_img1 = torch.from_numpy(np.transpose(hr_img1, (2, 0, 1))).to(
        device, non_blocking=True).unsqueeze(0).float() / 255.
    
    gt = torch.from_numpy(np.transpose(gt, (2, 0, 1))).to(
        device, non_blocking=True).unsqueeze(0).float() / 255.
    
    lr_img0 = torch.from_numpy(np.transpose(lr_img0, (2, 0, 1))).to(
        device, non_blocking=True).unsqueeze(0).float() / 255.
    lr_img1 = torch.from_numpy(np.transpose(lr_img1, (2, 0, 1))).to(
        device, non_blocking=True).unsqueeze(0).float() / 255.
    lr_img2 = torch.from_numpy(np.transpose(lr_img2, (2, 0, 1))).to(
        device, non_blocking=True).unsqueeze(0).float() / 255.
    
    hr_images = torch.cat((hr_img0, hr_img1), 1)
    
    images_LR_1_0 = img_to_flownet(lr_img1, lr_img0)
    images_LR_1_2 = img_to_flownet(lr_img1, lr_img2)
    imgs = torch.cat((lr_img0, lr_img1, lr_img2), 1)
    
    lr_images = torch.cat((images_LR_1_0, images_LR_1_2), 1)
    
    model = HSTRNet(device, args)
   
    result = model.inference(imgs, hr_images, lr_images, gt)
    
    psnr = -10 * math.log10(((gt - result) * (gt - result)).mean())
    print(psnr)
    
    
    
    result = result.cpu().detach().numpy()
    result = result[0, :]
    result = np.transpose(result, (1, 2, 0))
    cv2.imshow("win", result)
    cv2.waitKey(10000)
"""

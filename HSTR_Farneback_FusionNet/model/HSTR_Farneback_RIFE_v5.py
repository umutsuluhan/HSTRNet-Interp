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
        return list(self.unet.parameters())
    
    def convert(self, param):
        return {
        k.replace("module.", ""): v
            for k, v in param.items()
            if "module." in k
        }
    
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

    def inference(self, imgs, hr_images, lr_images, gt=None, training=False):
        
        lr_img0 = imgs[:, :3]
        lr_img1 = imgs[:, 3:6]
        lr_img2 = imgs[:, 6:9]
        
        images_LR_1_0 = lr_images[:, :3]
        images_LR_1_2 = lr_images[:, 3:6]
        
        start_time_rife = time.time()
        warped_hr_img1_0, warped_hr_img1_2, flow = self.ifnet(hr_images)     
        rife_time = time.time() - start_time_rife
        
        start_time_farneback = time.time()
        lr_F_1_0 = self.optical_flow_est(               
            torch.cat((lr_img1, lr_img0), 1))
        
        lr_F_1_2 = self.optical_flow_est(               
            torch.cat((lr_img1, lr_img2), 1))
        farneback_time = time.time() - start_time_farneback

        start_time_warp = time.time()
        warped_lr_img1_0 = warp(lr_img0, lr_F_1_0, self.device)
        warped_lr_img1_2 = warp(lr_img2, lr_F_1_2, self.device)
        warp_time = time.time() - start_time_warp

        # Try below code without times 2 
        flow = F.interpolate(flow, scale_factor=2.0, mode="bilinear",
                             align_corners=False) * 2.0
        
        start_time_context = time.time()
        c0_HR = self.contextnet(hr_images[:, :3], flow[:, :2])
        c1_HR = self.contextnet(hr_images[:, 3:6], flow[:, 2:4])
        c0_LR = self.contextnet(lr_img0, lr_F_1_0)
        c1_LR = self.contextnet(lr_img2, lr_F_1_2)
        context_time = time.time() - start_time_context

        start_time_fusion = time.time()        
        refine_output = self.unet(warped_lr_img1_0, lr_img1, warped_lr_img1_2, warped_hr_img1_0, warped_hr_img1_2, c0_HR, c1_HR, c0_LR, c1_LR)
        fusion_time = time.time() - start_time_fusion
        
        res = torch.sigmoid(refine_output[:, :3]) * 2 - 1
        mask = torch.sigmoid(refine_output[:, 3:4])
        merged_img = warped_hr_img1_0 * mask + warped_hr_img1_2 * (1 - mask)
        pred = merged_img + res
        pred = torch.clamp(pred, 0, 1)

        return pred, rife_time, farneback_time, warp_time, context_time, fusion_time

        
def img_to_flownet(img0, img1):
    images = torch.cat((img0, img1), 0)
    images = images.permute(1, 0, 2, 3)
    images = images.unsqueeze(0).cuda()
    return images
    
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

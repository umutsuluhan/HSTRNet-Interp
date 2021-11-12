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

from model.flownet2.models import FlowNet2
from model.RIFE_v5.warplayer import warp
from model.RIFE_v5.IFNet import IFNet
from model.RIFE_v5.RIFE import ContextNet, FusionNet



class HSTRNet:
    def __init__(self, device, args):
        self.device = "cpu"
        
        self.ifnet = IFNet()
        self.flownet2 = FlowNet2(args)
        self.contextnet = ContextNet()
        self.unet = FusionNet()
        
        self.ifnet.to(self.device)
        self.flownet2 = self.flownet2.cuda()
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

    def inference(self, imgs, hr_images, lr_images, gt=None, training=False):
        
        
        lr_img0 = imgs[:, :3]
        lr_img1 = imgs[:, 3:6]
        lr_img2 = imgs[:, 6:9]
        
        images_LR_1_0 = lr_images[:, :3]
        images_LR_1_2 = lr_images[:, 3:6]
        
        # state_dict = self.convert(torch.load('/home/hus/Desktop/repos/HSTRNet/HSTR_Flownet2/model/RIFE_v5/train_log.large/unet.pkl', map_location=self.device))
        # pretrained_dict = {}
        # for k, v in zip(state_dict, model.unet.named_parameters()):
        #     # print(k)
        #     # print(state_dict[k].shape)
        #     # print(v[1].shape)
        #     # print(state_dict[k].shape == v[1].shape)
        #     if state_dict[k].shape == v[1].shape:
        #         pretrained_dict[k] = state_dict[k]
                
        
        # self.unet.load_state_dict(pretrained_dict, strict=False)

        warped_hr_img1_0, warped_hr_img1_2, flow = self.ifnet(hr_images)        
        warped_hr_img1_0 = warped_hr_img1_0.to(self.device)
        warped_hr_img1_2 = warped_hr_img1_2.to(self.device)
        
        lr_F_1_0 = self.flownet2(images_LR_1_0)
        lr_F_1_2 = self.flownet2(images_LR_1_2)
        
        lr_F_1_0 = lr_F_1_0.to(self.device)
        lr_F_1_2 = lr_F_1_2.to(self.device)
        

        warped_lr_img1_0 = warp(lr_img0, lr_F_1_0, self.device)
        warped_lr_img1_2 = warp(lr_img2, lr_F_1_2, self.device)
        
        flow = F.interpolate(flow, scale_factor=2.0, mode="bilinear",
                             align_corners=False) * 2.0
        
        
        c0_HR = self.contextnet(hr_images[:, :3], flow[:, :2])
        c1_HR = self.contextnet(hr_images[:, 3:6], flow[:, 2:4])
        
        c0_LR = self.contextnet(lr_img0, lr_F_1_0)
        c1_LR = self.contextnet(lr_img2, lr_F_1_2)

        refine_output = self.unet(warped_lr_img1_0, lr_img1, warped_lr_img1_2, warped_hr_img1_0, warped_hr_img1_2, c0_HR, c1_HR, c0_LR, c1_LR)
        res = torch.sigmoid(refine_output[:, :3]) * 2 - 1
        mask = torch.sigmoid(refine_output[:, 3:4])
        merged_img = warped_hr_img1_0 * mask + warped_hr_img1_2 * (1 - mask)
        pred = merged_img + res
        pred = torch.clamp(pred, 0, 1)

        return pred

        
def img_to_flownet(img0, img1):
    images = torch.cat((img0, img1), 0)
    images = images.permute(1, 0, 2, 3)
    images = images.unsqueeze(0).cuda()
    return images
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fp16",
        action="store_true",
        default="False",
        help="Run model in pseudo-fp16 mode (fp16 storage fp32 math).",
    )
    parser.add_argument("--rgb_max", type=float, default=255.0)

    args = parser.parse_args()

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

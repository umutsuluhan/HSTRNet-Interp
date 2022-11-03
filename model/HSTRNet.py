import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.networks.warplayer import warp
from model.networks.IFNet import IFNet
from model.networks.RIFE import ContextNet, FusionNet

from utils.utils import convert

class HSTRNet(nn.Module):
    def __init__(self, device, warping):
        super(HSTRNet, self).__init__()
        self.device = device
        
        self.warping = warping
        self.ifnet = IFNet()
        self.contextnet = ContextNet(warping)
        self.unet = FusionNet()
        
        self.ifnet.to(self.device)
        self.contextnet.to(self.device)
        self.unet.to(self.device)
        
    def load_pretrained(self, path):
        self.ifnet.load_state_dict(convert(torch.load(os.path.join(path, "ifnet.pkl"), map_location=self.device)))
        if not self.warping:
            self.contextnet.load_state_dict(torch.load(os.path.join(path, "contextnet.pkl"), map_location=self.device))
        else:
            self.contextnet.load_state_dict(convert(torch.load(os.path.join(path, "contextnet.pkl"), map_location=self.device)))
        self.unet.load_state_dict(torch.load(os.path.join(path, "unet.pkl"), map_location=self.device))
    
    def train(self):
        self.ifnet.train()
        self.contextnet.train()
        self.unet.train()

    def eval(self):
        self.ifnet.eval()
        self.contextnet.eval()
        self.unet.eval()

    def freeze(self, module):
        for k, v in module.named_parameters():
            v.requires_grad = False
        return module
    
    def forward(self, lr_images, hr_images):
        assert lr_images.shape[1] == 9, "Dimension match failure. You should provide 3 LR images concatenated with respect to 1st dimension."
        assert hr_images.shape[1] == 6, "Dimension match failure. You should provide 2 HR images concatenated with respect to 1st dimension." 
        assert lr_images.shape[2] == hr_images.shape[2], "Height match failure. LR images and HR images must have same height attribute."
        assert lr_images.shape[3] == hr_images.shape[3], "Width match failure. LR images and HR images must have same width attribute."
        
        lr_img0 = lr_images[:, :3]
        lr_img1 = lr_images[:, 3:6]
        lr_img2 = lr_images[:, 6:9]
        
        warped_hr_img1_0, warped_hr_img1_2, flow = self.ifnet(hr_images)   
        
        _, _, lr_F_1_0_ = self.ifnet(torch.cat((lr_img1, lr_img0),1)) 
        _, _, lr_F_1_2_ = self.ifnet(torch.cat((lr_img1, lr_img2),1)) 
        
        if not self.warping:
            lr_F_1_0 = lr_F_1_0_ * -2.0 
            lr_F_1_2 = lr_F_1_2_ * 2.0 
        else:
            lr_F_1_0 = lr_F_1_0_ * 2
            lr_F_1_2 = lr_F_1_2_ * 2

        lr_F_1_0 = F.interpolate(lr_F_1_0, scale_factor=2.0, mode="bilinear",align_corners=False) * 2.0
        lr_F_1_2 = F.interpolate(lr_F_1_2, scale_factor=2.0, mode="bilinear", align_corners=False) * 2.0
        
        if not self.warping:
            warped_lr_img1_0 = warp(lr_img0, lr_F_1_0[:, :2], self.device) 
            warped_lr_img1_2 = warp(lr_img2, lr_F_1_2[:, 2:4], self.device) 
        else:
            #warped_lr_img1_0 = warp(lr_img0, lr_F_1_0, self.device) 
            #warped_lr_img1_2 = warp(lr_img2, lr_F_1_2, self.device) 
            warped_lr_img1_0 = warp(lr_img0, lr_F_1_0[:, 2:4], self.device) 
            warped_lr_img1_2 = warp(lr_img2, lr_F_1_2[:, 2:4], self.device) 
        
        if self.warping:
            flow = F.interpolate(flow, scale_factor=2.0, mode="bilinear", align_corners=False) * 2.0

        c0_HR = self.contextnet(hr_images[:, :3], flow[:, :2])
        c1_HR = self.contextnet(hr_images[:, 3:6], flow[:, 2:4])
        if not self.warping:
            c0_LR = self.contextnet(lr_img0, lr_F_1_0_[:, :2])
            c1_LR = self.contextnet(lr_img2, lr_F_1_2_[:, 2:4])
        else:
            #c0_LR = self.contextnet(lr_img0, lr_F_1_0)
            #c1_LR = self.contextnet(lr_img2, lr_F_1_2)
            c0_LR = self.contextnet(lr_img0, lr_F_1_0[:, 2:4])
            c1_LR = self.contextnet(lr_img2, lr_F_1_2[:, 2:4])

        refine_output = self.unet(warped_lr_img1_0, lr_img1, warped_lr_img1_2, warped_hr_img1_0, warped_hr_img1_2, c0_HR, c1_HR, c0_LR, c1_LR)
        res = torch.sigmoid(refine_output[:, :3]) * 2 - 1
        mask = torch.sigmoid(refine_output[:, 3:4])
        merged_img = warped_hr_img1_0 * mask + warped_hr_img1_2 * (1 - mask)
        pred = merged_img + res
        pred = torch.clamp(pred, 0, 1)

        return pred
# Model implementation of Farneback + Super_Slomo

import torch
import torch.nn as nn
import cv2
import time
import math
import numpy as np
import sys
import argparse
import torch.nn.functional as F
from flownet2.models import FlowNet2
from flownet2.utils.frame_utils import read_gen
from warplayer import warp
from refine import Contextnet, Unet
from IFNet import IFNet


class HSTR_FSS:
    def __init__(self, device, args):
        self.device = "cpu"
        
        self.ifnet = IFNet()
        self.ifnet.to(device)
        self.flownet2 = FlowNet2(args).cuda()
        
        self.contextnet = Contextnet()
        self.unet = Unet()
        
        
    def return_parameters(self):
        return list(self.unet.parameters())
    
    def convert(self, param):
        return {
        k.replace("module.", ""): v
            for k, v in param.items()
            if "module." in k
        }
        
    def np_to_torch(self, img):
        result = (
            torch.from_numpy(np.transpose(img, (2, 0, 1)))
            .to(device, non_blocking=True)
            .unsqueeze(0)
            .float()
            / 255.0
        )
        
        return result

    def inference(self, imgs, hr_images, lr_images, gt, training=False):
        
        images_LR_1_0 = lr_images[:, :3]  # lr_img0 = lr(t-1)
        images_LR_1_2 = lr_images[:, 3:6]  # lr_img1 = lr(t)
        
        self.ifnet.load_state_dict(
            self.convert(torch.load('/home/hus/Desktop/repos/HSTRNet/HSTR_Farneback_SuperSlomo/trained_model/train_log/flownet.pkl', map_location=self.device)))

        state_dict = torch.load('/home/hus/Desktop/repos/HSTRNet/HSTR_Farneback_SuperSlomo/trained_model/train_log/flownet.pkl')
        
        for k in state_dict:
            print(k)

        dict = torch.load("/home/hus/Desktop/repos/HSTRNet/HSTR_Flownet2/trained_models/FlowNet2_checkpoint.pth.tar")
        self.flownet2.load_state_dict(dict["state_dict"])
        self.flownet2.eval()

        warped_hr_img1_0, warped_hr_img1_2 = self.ifnet(hr_images)
        warped_hr_img1_0 = warped_hr_img1_0.to(device)
        warped_hr_img1_2 = warped_hr_img1_2.to(device)
        
        lr_F_1_0 = self.flownet2(images_LR_1_0)
        # Try times 2 flows
        lr_F_1_2 = self.flownet2(images_LR_1_2)

        warped_lr_img1_0 = warp(lr_img0, lr_F_1_0, self.device)
        warped_lr_img1_2 = warp(lr_img2, lr_F_1_2, self.device)
        
        # psnr = -10 * math.log10(((gt - warped_hr_img1_0) * (gt - warped_hr_img1_0)).mean())
        # print(psnr)
        
        # psnr = -10 * math.log10(((gt - warped_hr_img1_2) * (gt - warped_hr_img1_2)).mean())
        # print(psnr)
        
        # psnr = -10 * math.log10(((lr_img1 - warped_lr_img1_0) * (lr_img1 - warped_lr_img1_0)).mean())
        # print(psnr)
        
        # psnr = -10 * math.log10(((lr_img1 - warped_lr_img1_2) * (lr_img1 - warped_lr_img1_2)).mean())
        # print(psnr)

        input_imgs = torch.cat(
            (warped_lr_img1_0, lr_img1, warped_lr_img1_2, warped_hr_img1_0, warped_hr_img1_0), dim=1
        )

        result_frame = self.unet(input_imgs)

        return result_frame

        # if training == False:
            # return result_frame
        # else:
        #     return (
        #         result,
        #         g_I0_F_t_0,
        #         g_I1_F_t_1,
        #         warped_lr_img1_0,
        #         lr_img0,
        #         warped_lr_img1_2,
        #         lr_img2,
        #     )
def img_to_flownet(img0, img1):
    images = [img0, img1]
    images = np.array(images).transpose(3, 0, 1, 2)
    im = torch.from_numpy(images.astype(np.float32)).unsqueeze(0).cuda()
    return im

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
    
    hr_images = torch.cat((hr_img0, hr_img1), 1)
    
    images_LR_1_0 = img_to_flownet(lr_img1, lr_img0)
    images_LR_1_2 = img_to_flownet(lr_img1, lr_img2)
    
    lr_img0 = torch.from_numpy(np.transpose(lr_img0, (2, 0, 1))).to(
        device, non_blocking=True).unsqueeze(0).float() / 255.
    lr_img1 = torch.from_numpy(np.transpose(lr_img1, (2, 0, 1))).to(
        device, non_blocking=True).unsqueeze(0).float() / 255.
    lr_img2 = torch.from_numpy(np.transpose(lr_img2, (2, 0, 1))).to(
        device, non_blocking=True).unsqueeze(0).float() / 255.
    
    imgs = torch.cat((lr_img0, lr_img1, lr_img2), 1)
    
    lr_images = torch.cat((images_LR_1_0, images_LR_1_2), 1)
    
    model = HSTR_FSS(device, args)
   

    result = model.inference(imgs, hr_images, lr_images, gt).cpu().detach().numpy()
    # result = result[0, :]
    # result = np.transpose(result, (1, 2, 0))
    # cv2.imshow("win", result)
    # cv2.waitKey(10000)

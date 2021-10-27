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
from flownet2.models import FlowNet2
#from pytorch_msssim import ssim
from flownet2.utils.frame_utils import read_gen


class HSTR_FSS:
    def __init__(self, device, args):
        self.device = device
        self.unet = UNet(15, 3)
        self.flownet = FlowNet2(args).cuda()
        # self.flownet.to(device)
        self.unet.to(self.device)

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

    def intermediate_flow_est(self, x, t):

        F_0_1 = x[:, :2].detach().cpu().numpy()
        F_1_0 = x[:, 2:4].detach().cpu().numpy()

        temp = -t * (1 - t)
        fCoeff = [temp, t * t, (1 - t) * (1 - t), temp]

        F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
        F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0

        return F_t_0, F_t_1

    def img_to_flownet(self, img0, img1):
        images = [img0, img1]
        images = np.array(images).transpose(3, 0, 1, 2)
        im = torch.from_numpy(images.astype(np.float32)).unsqueeze(0).cuda()
        return im
        
    def np_to_torch(self, img):
        result = (
            torch.from_numpy(np.transpose(img, (2, 0, 1)))
            .to(device, non_blocking=True)
            .unsqueeze(0)
            .float()
            / 255.0
        )
        
        return result
        
    

    def inference(self, imgs, timestamps, training=False):

        # t = 0.5                        # Timestamp of the generated frame.

        hr_img0 = imgs[:, :, :3]  # hr_img0 = hr(t-1)
        hr_img1 = imgs[:, :, 3:6]  # hr_img1 = hr(t+1)
        lr_img0 = imgs[:, :, 6:9]  # lr_img0 = lr(t-1)
        lr_img1 = imgs[:, :, 9:12]  # lr_img1 = lr(t)
        lr_img2 = imgs[:, :, 12:15]  # lr_img2 = lr(t+1)
        
        dict = torch.load("/home/hus/Desktop/repos/HSTRNet/HSTR_Flownet2/trained_models/FlowNet2_checkpoint.pth.tar")
        self.flownet.load_state_dict(dict["state_dict"])
        self.flownet.eval()
        
        images_HR_0_1 = self.img_to_flownet(hr_img0, hr_img1)
        hr_F_0_1 = self.flownet(images_HR_0_1)
        hr_F_0_1 = hr_F_0_1 * 2
        
        images_HR_1_0 = self.img_to_flownet(hr_img1, hr_img0)
        hr_F_1_0 = self.flownet(images_HR_1_0)
        hr_F_1_0 = hr_F_1_0 * 2
        
        images_LR_1_0 = self.img_to_flownet(lr_img1, lr_img0)
        lr_F_1_0 = self.flownet(images_LR_1_0)
        lr_F_1_0 = lr_F_1_0 * 2
        
        images_LR_1_2 = self.img_to_flownet(lr_img1, lr_img2)
        lr_F_1_2 = self.flownet(images_LR_1_2)
        lr_F_1_2 = lr_F_1_2 * 2
        
        
        F_t_0, F_t_1 = self.intermediate_flow_est(  # Flow from t to 0 and flow from t to 1 using provided low fps video frames
            torch.cat((hr_F_0_1, hr_F_1_0), 1), 0.5)

        F_t_0 = torch.from_numpy(F_t_0).to(self.device)
        F_t_1 = torch.from_numpy(F_t_1).to(self.device)

        hr_img0 = self.np_to_torch(hr_img0)
        hr_img1 = self.np_to_torch(hr_img1)
        
        lr_img0 = self.np_to_torch(lr_img0)
        lr_img1 = self.np_to_torch(lr_img1)
        lr_img2 = self.np_to_torch(lr_img2)
        
        # Backwarping module
        backwarp = backWarp(hr_img0.shape[3], hr_img0.shape[2], self.device)

        # I0  = backwarp(I1, F_0_1)

        # Backwarp of I0 and F_t_0
        g_I0_F_t_0 = backwarp(hr_img0, F_t_0)
        # Backwarp of I1 and F_t_1
        g_I1_F_t_1 = backwarp(hr_img1, F_t_1)

        MSE_loss_fn = nn.MSELoss()

        MSE_loss = MSE_loss_fn(g_I0_F_t_0, hr_img0)

        result = g_I0_F_t_0.cpu().detach().numpy()
        result = result[0, :]
        result = np.transpose(result, (1, 2, 0))
        cv2.imshow("win", result)
        cv2.waitKey(10000)

        result = hr_img0.cpu().detach().numpy()
        result = result[0, :]
        result = np.transpose(result, (1, 2, 0))
        cv2.imshow("win1", result)
        cv2.waitKey(10000)

        psnr1 = float((10 * math.log10(1 / MSE_loss.item())))
#        ssim1 = float(ssim(g_I0_F_t_0, hr_img0))

        MSE_loss = MSE_loss_fn(g_I1_F_t_1, hr_img1)
        psnr2 = float((10 * math.log10(1 / MSE_loss.item())))
#        ssim2 = float(ssim(g_I1_F_t_1, hr_img1))

        print(psnr1)
#        print(ssim1)
        print(psnr2)
#       print(ssim2)

        warped_lr_img1_0 = backwarp(lr_img0, lr_F_1_0)
        warped_lr_img1_2 = backwarp(lr_img2, lr_F_1_2)

        input_imgs = torch.cat(
            (warped_lr_img1_0, lr_img1, warped_lr_img1_2, g_I0_F_t_0, g_I1_F_t_1), dim=1
        )

        Ft_p = self.unet(input_imgs)

        result = Ft_p

        if training == False:
            return result
        else:
            return (
                result,
                g_I0_F_t_0,
                g_I1_F_t_1,
                warped_lr_img1_0,
                lr_img0,
                warped_lr_img1_2,
                lr_img2,
            )


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

    img0_HR = read_gen(
        "/home/hus/Desktop/data/vimeo_triplet/sequences/00001/0001/im1.png"
    )

    img1_HR = read_gen(
        "/home/hus/Desktop/data/vimeo_triplet/sequences/00001/0001/im3.png"
    )

    img0_LR = read_gen(
        "/home/hus/Desktop/data/vimeo_triplet/sequences/00001/0001/im1.png"
    )
    img1_LR = read_gen(
        "/home/hus/Desktop/data/vimeo_triplet/sequences/00001/0001/im2.png"
    )
    img2_LR = read_gen(
        "/home/hus/Desktop/data/vimeo_triplet/sequences/00001/0001/im3.png"
    )

    """padding1_mult = math.floor(img0_HR.shape[0] / 32) + 1
    padding2_mult = math.floor(img0_HR.shape[1] / 32) + 1
    pad1 = (32 * padding1_mult) - img0_HR.shape[0]
    pad2 = (32 * padding2_mult) - img0_HR.shape[1]

    # Padding to meet dimension requirements of the network
    # Done before network call, otherwise slows down the network
    padding1 = nn.ReplicationPad2d((0, pad2, pad1, 0))"""

    device = "cpu"

    imgs = np.concatenate((img0_HR, img1_HR, img0_LR, img1_LR, img2_LR), 2)
    # imgs = padding1(imgs)
    model = HSTR_FSS(device, args)
    # model.eval()

    # result = model.inference(imgs, []).cpu().detach().numpy()
    # result = result[0, :]
    # result = np.transpose(result, (1, 2, 0))
    # cv2.imshow("win", result)
    # cv2.waitKey(10000)

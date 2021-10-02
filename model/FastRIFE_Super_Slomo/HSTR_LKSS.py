# Model implementation of Lucas Kanade + Super_Slomo

import torch
import torch.nn as nn
import cv2
import time
import math
import numpy as np
import sys
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from .unet_model import UNet
from .backwarp import backWarp

device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

class HSTR_LKSS():

    def __init__(self):
        self.unet = UNet(4, 32)
        self.arbitrary_time_f_int = UNet(15, 3)
        self.arbitrary_time_f_int.to(device)
        self.unet.to(device)

    def optical_flow_est(self, x):
        x = F.interpolate(x, scale_factor=0.5, mode="bilinear",
                          align_corners=False)
        img0 = x[:, :3].cpu().numpy()
        img1 = x[:, 3:].cpu().numpy()

        feature_params = dict(maxCorners=100,
                              qualityLevel=0.1,
                              minDistance=10,
                              blockSize=7)

        lk_params = dict(winSize=(15, 15),
                         maxLevel=0,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        num_samples, _, x, y = img0.shape
        flow_batch = np.empty((0, 2, x, y))
        flow_time = []
        for i in range(num_samples):
            img0_single = img0[i, :, :, :].reshape(x, y, 3)
            img1_single = img1[i, :, :, :].reshape(x, y, 3)
            img0_single = cv2.cvtColor(img0_single, cv2.COLOR_BGR2GRAY)
            img1_single = cv2.cvtColor(img1_single, cv2.COLOR_BGR2GRAY)

            start2 = time.time()
            p0 = cv2.goodFeaturesToTrack(img0_single, mask=None, **feature_params)
            if p0 is None:
                p0 = np.ones((1, 1, 2)).astype(np.float32)

            img0_single = np.uint8(img0_single * 255.0)
            img1_single = np.uint8(img1_single * 255.0)
            flow_single, _st, _err = cv2.calcOpticalFlowPyrLK(img0_single, img1_single, p0, None, **lk_params)
            end2 = time.time()

            p0 = p0.reshape(-1, 2)
            flow_x = csr_matrix((flow_single[:,:,0].reshape(-1), (p0[:,1], p0[:,0])), shape=(x, y)).toarray().reshape(1, x, y)
            flow_y = csr_matrix((flow_single[:,:,1].reshape(-1), (p0[:,1], p0[:,0])), shape=(x, y)).toarray().reshape(1, x, y)
            flows = np.append(flow_x, flow_y, axis=0)

            flow_time.append((end2 - start2) * 1000)
            flow_single = flows.reshape(1, 2, x, y)
            flow_batch = np.append(flow_batch, flow_single, axis=0)
        return torch.tensor(flow_batch, dtype=torch.float)

    def intermediate_flow_est(self, x, t):

        F_0_1 = x[:, :2].cpu().numpy()
        F_1_0 = x[:, 2:4].cpu().numpy()

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
        
        # warped_lfr_img0 = backwarp(lfr_img1, lfr_F_0_1)     # Backwarp of LFR_I0 and F_t_0
        # warped_lfr_img1 = backwarp(lfr_img0, lfr_F_1_0)     # Backwarp of LFR_I1 and F_t_0
        
        warped_hfr_img0 = backwarp(hfr_img1, hfr_F_0_1)     # Backwarp of HFR_I0 and F_t_0
        # result1 = warped_hfr_img0.detach().numpy()
        # result1 = result1[0,:]
        # result1 = np.transpose(result1, (1, 2, 0))
        # print(result1.shape)
        # cv2.imshow("win", result1)
        # cv2.waitKey(1000)
        
        #warped_hfr_img1 = backwarp(hfr_img0, hfr_F_1_0)     # Backwarp of HFR_I1 and F_t_0
        warped_hfr_img2 = backwarp(hfr_img1, hfr_F_2_1)     # Backwarp of HFR_I2 and F_t_0

        # Interpolation of flows to match tensor sizes

        # lfr_F_0_1 = F.interpolate(lfr_F_0_1, scale_factor=2, mode="bilinear",    
        #                      align_corners=False) 
        # lfr_F_1_0 = F.interpolate(lfr_F_1_0, scale_factor=2, mode="bilinear",   
        #                      align_corners=False) 
        F_t_1 = F.interpolate(F_t_1, scale_factor=2, mode="bilinear", 
                             align_corners=False)
        F_t_0 = F.interpolate(F_t_0, scale_factor=2, mode="bilinear",  
                             align_corners=False)

        
        input_imgs = torch.cat((warped_hfr_img0,  hfr_img1,warped_hfr_img2, g_I0_F_t_0, g_I1_F_t_1), dim=1)
        
        padding1_mult= math.floor(input_imgs.shape[2] / 32) + 1
        padding2_mult= math.floor(input_imgs.shape[3] / 32) + 1
        pad1 = (32 * padding1_mult) - input_imgs.shape[2]
        pad2 = (32 * padding2_mult) - input_imgs.shape[3]
        
        # Padding to meet dimension requirements of the network
        # Done before network call, otherwise slows down the network
        padding1 = nn.ReplicationPad2d((0, pad2, pad1, 0))       
        
        input_imgs = padding1(input_imgs)
        
        
        Ft_p = self.arbitrary_time_f_int(input_imgs)

        padding2 = nn.ReplicationPad2d((0, -pad2, -pad1, 0)) 
        Ft_p = padding2(Ft_p)

        # F_t_0_f = intrpOut[:, :2, :, :] + F_t_0
        # F_t_1_f = intrpOut[:, 2:4, :, :] + F_t_1
        # V_t_0   = torch.sigmoid(intrpOut[:, 4:5, :, :])
        # V_t_1   = 1 - V_t_0

        # g_I0_F_t_0_f = backwarp(lfr_img0, F_t_0_f)
        # g_I1_F_t_1_f = backwarp(lfr_img1, F_t_1_f)

        # wCoeff = [1 - t, t]

        # Ft_p = (wCoeff[0] * V_t_0 * g_I0_F_t_0_f + wCoeff[1] * V_t_1 * g_I1_F_t_1_f) / (wCoeff[0] * V_t_0 + wCoeff[1] * V_t_1)
        
        result = Ft_p.detach().numpy()
        result = result[0,:]
        result = np.transpose(result, (1, 2, 0))
        cv2.imshow("win", result)
        cv2.waitKey(100)
        
        return result

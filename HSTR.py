import torch
import torch.nn as nn
import numpy as np
from model.IFNet import IFNet
from model.warplayer import warp
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model():
    
    def __init__(self):
        self.flownet = IFNet()
        self.flownet.to(device)
        
    def inference(self, imgs, timestamps,training=False):
        
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
        lfr_img0 = torch.from_numpy(np.transpose(lfr_img0, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
        lfr_img1 = torch.from_numpy(np.transpose(lfr_img1, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
        hfr_img0 = torch.from_numpy(np.transpose(hfr_img0, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
        hfr_img1 = torch.from_numpy(np.transpose(hfr_img1, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
        hfr_img2 = torch.from_numpy(np.transpose(hfr_img2, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
        
        lfr_prev_to_next_flow,  _ = self.flownet(torch.cat((lfr_img0, lfr_img1),1))
        lfr_next_to_prev_flow,  _ = self.flownet(torch.cat((lfr_img1, lfr_img0),1))
        
        hfr_prev_to_t_flow, _ = self.flownet(torch.cat((hfr_img0, hfr_img1),1))
        hfr_t_to_next_flow, _ = self.flownet(torch.cat((hfr_img1, hfr_img2),1))
        
            
            
            
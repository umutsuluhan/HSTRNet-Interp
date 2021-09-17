import torch
import torch.nn as nn
from model.IFNet import IFNet
from model.warplayer import warp
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model():
    
    def __init__(self):
        self.flownet = IFNet()
        self.flownet.to(device)
        
    def inference(self, imgs, training=False):
        
        # lfr_img0 = lr(t-1)
        # lfr_img1 = lr(t)
        # lfr_img2 = lr(t+1)
        # hfr_img0 = hr(t-1)
        # hfr_img1 = hr(t+1)
        
        lfr_img0 = imgs[:, :3]
        lfr_img1 = imgs[:, 3:6]
        lfr_img2 = imgs[:, 6:9]
        hfr_img0 = imgs[:, 9:12]
        hfr_img1 = imgs[:, 12:15]
        
        hfr_prev_to_next_flow, _ = self.flownet(torch.cat((hfr_img0, hfr_img1),1))
        hfr_next_to_prev_flow, _ = self.flownet(torch.cat((hfr_img1, hfr_img0),1))
        lfr_prev_to_mid_flow,  _ = self.flownet(torch.cat((lfr_img0, lfr_img1),1))
        lfr_mid_to_next_flow,  _ = self.flownet(torch.cat((lfr_img1, lfr_img2),1))
        
        # print(hfr_prev_to_next_flow.shape)
        # print(hfr_next_to_prev_flow.shape)
        # print(lfr_prev_to_mid_flow.shape)
        # print(lfr_mid_to_next_flow.shape)
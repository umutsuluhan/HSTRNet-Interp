import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
import torch.optim as optim
import itertools
from model.warplayer import warp
from torch.nn.parallel import DistributedDataParallel as DDP
from model.IFNet import *
import torch.nn.functional as F
from model.loss import *
import cv2
import math

device = torch.device("cuda")
    
class Model:
    def __init__(self, local_rank=-1):
        self.flownet = IFNet()
        self.device()
        self.optimG = AdamW(self.flownet.parameters(), lr=1e-6, weight_decay=1e-3) # use large weight decay may avoid NaN loss
        self.epe = EPE()
        #self.lap = LapLoss()
        self.sobel = SOBEL()
        if local_rank != -1:
            self.flownet = DDP(self.flownet, device_ids=[local_rank], output_device=local_rank)

    def train(self):
        self.flownet.train()

    def eval(self):
        self.flownet.eval()

    def device(self):
        self.flownet.to(device)

    def load_model(self, path, rank=0):
        def convert(param):
            return {
            k.replace("module.", ""): v
                for k, v in param.items()
                if "module." in k
            }
            
        if rank <= 0:
            self.flownet.load_state_dict(convert(torch.load('{}/flownet.pkl'.format(path))))
        
    def save_model(self, path, rank=0):
        if rank == 0:
            torch.save(self.flownet.state_dict(),'{}/flownet.pkl'.format(path))

    def inference(self, img0, img1, scale_list=[4, 2, 1], TTA=False):
        imgs = torch.cat((img0, img1), 1)
        flow, mask, merged, flow_teacher, merged_teacher, loss_distill = self.flownet(imgs, scale_list)
        if TTA == False:
            return merged[2]
        else:
            flow2, mask2, merged2, flow_teacher2, merged_teacher2, loss_distill2 = self.flownet(imgs.flip(2).flip(3), scale_list)
            return (merged[2] + merged2[2].flip(2).flip(3)) / 2
    
    def update(self, imgs, gt, learning_rate=0, mul=1, training=True, flow_gt=None):
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate
        img0 = imgs[:, :3]
        img1 = imgs[:, 3:]
        if training:
            self.train()
        else:
            self.eval()
        flow, mask, merged, flow_teacher, merged_teacher, loss_distill = self.flownet(torch.cat((imgs, gt), 1), scale=[4, 2, 1])
        loss_l1 = (self.lap(merged[2], gt)).mean()
        loss_tea = (self.lap(merged_teacher, gt)).mean()
        if training:
            self.optimG.zero_grad()
            loss_G = loss_l1 + loss_tea + loss_distill * 0.01
            loss_G.backward()
            self.optimG.step()
        else:
            flow_teacher = flow[2]
            merged_teacher = merged[2] 
        return merged[2], {
            'merged_tea': merged_teacher,
            'mask': mask,
            'mask_tea': mask,
            'flow': flow[2][:, :2],
            'flow_tea': flow_teacher,
            'loss_l1': loss_l1,
            'loss_tea': loss_tea,
            'loss_distill': loss_distill,
            }


if __name__ == '__main__':
    img0_HR = cv2.imread(
        "/home/hus/Desktop/data/vimeo_triplet/sequences/00001/0412/im1.png")
    gt = cv2.imread(
        "/home/hus/Desktop/data/vimeo_triplet/sequences/00001/0412/im2.png")
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

    img0_HR = torch.from_numpy(np.transpose(img0_HR, (2, 0, 1))).to(
        device, non_blocking=True).unsqueeze(0).float() / 255.
    gt = torch.from_numpy(np.transpose(gt, (2, 0, 1))).to(
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
    model = Model()
    model.load_model("/home/hus/Desktop/repos/HSTRNet/RIFE_HSTR_v6/train_log")
    model.eval()

    result = model.inference(imgs)
    
    psnr = -10 * math.log10(((gt - result) * (gt - result)).mean())
    print(psnr)

    result = result.cpu().detach().numpy()
    result = result[0, :]
    result = np.transpose(result, (1, 2, 0))
    cv2.imshow("win", result)
    cv2.waitKey(2000)
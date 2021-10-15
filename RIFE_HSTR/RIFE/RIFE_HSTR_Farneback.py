import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
import torch.optim as optim
import itertools
from warplayer import warp
from torch.nn.parallel import DistributedDataParallel as DDP
from scipy.sparse import csr_matrix
from IFNet import *
import torch.nn.functional as F
from loss import *
import cv2
import math
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes,
                                 kernel_size=4, stride=2, padding=1, bias=True),
        nn.PReLU(out_planes)
    )


def conv_woact(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
    )


class Conv2(nn.Module):
    def __init__(self, in_planes, out_planes, stride=2):
        super(Conv2, self).__init__()
        self.conv1 = conv(in_planes, out_planes, 3, stride, 1)
        self.conv2 = conv(out_planes, out_planes, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


c = 16


class ContextNet(nn.Module):
    def __init__(self):
        super(ContextNet, self).__init__()
        self.conv1 = Conv2(3, c)
        self.conv2 = Conv2(c, 2*c)
        self.conv3 = Conv2(2*c, 4*c)
        self.conv4 = Conv2(4*c, 8*c)

    def forward(self, x, flow):
        x = self.conv1(x)
        f1 = warp(x, flow)
        x = self.conv2(x)
        flow = F.interpolate(flow, scale_factor=0.5,
                             mode="bilinear", align_corners=False) * 0.5
        f2 = warp(x, flow)
        x = self.conv3(x)
        flow = F.interpolate(flow, scale_factor=0.5,
                             mode="bilinear", align_corners=False) * 0.5
        f3 = warp(x, flow)
        x = self.conv4(x)
        flow = F.interpolate(flow, scale_factor=0.5,
                             mode="bilinear", align_corners=False) * 0.5
        f4 = warp(x, flow)
        return [f1, f2, f3, f4]


class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()
        self.down0 = Conv2(15, 2*c)
        self.down1 = Conv2(4*c, 4*c)
        self.down2 = Conv2(8*c, 8*c)
        self.down3 = Conv2(16*c, 16*c)
        self.up0 = deconv(32*c, 8*c)
        self.up1 = deconv(16*c, 4*c)
        self.up2 = deconv(8*c, 2*c)
        self.up3 = deconv(4*c, c)
        self.conv = nn.Conv2d(c, 4, 3, 1, 1)

    def forward(self, img0_HR, img1_HR, img0_LR, img1_LR, img2_LR, flow_HR, flow_LR_0_1, flow_LR_2_1, c0_HR, c1_HR):
        warped_img0_HR = warp(img0_HR, flow_HR[:, :2])
        warped_img1_HR = warp(img1_HR, flow_HR[:, 2:4])

        # Check below again
        warped_img0_LR = warp(img0_LR, flow_LR_0_1)
        warped_img2_LR = warp(img2_LR, flow_LR_2_1)

        s0 = self.down0(torch.cat((warped_img0_HR, warped_img1_HR,
                        warped_img0_LR, warped_img2_LR, img1_LR), 1))
        s1 = self.down1(
            torch.cat((s0, c0_HR[0], c1_HR[0]), 1))
        s2 = self.down2(
            torch.cat((s1, c0_HR[1], c1_HR[1]), 1))
        s3 = self.down3(
            torch.cat((s2, c0_HR[2], c1_HR[2]), 1))
        x = self.up0(
            torch.cat((s3, c0_HR[3], c1_HR[3]), 1))
        x = self.up1(torch.cat((x, s2), 1))
        x = self.up2(torch.cat((x, s1), 1))
        x = self.up3(torch.cat((x, s0), 1))
        x = self.conv(x)
        return x, warped_img0_HR, warped_img1_HR, warped_img0_LR, warped_img2_LR


class Model:
    def __init__(self, local_rank=-1):
        self.flownet = IFNet()
        self.contextnet = ContextNet()
        self.fusionnet = FusionNet()
        self.device()
        self.optimG = AdamW(itertools.chain(
            self.flownet.parameters(),
            self.contextnet.parameters(),
            self.fusionnet.parameters()), lr=1e-6, weight_decay=1e-4)
        self.schedulerG = optim.lr_scheduler.CyclicLR(
            self.optimG, base_lr=1e-6, max_lr=1e-3, step_size_up=8000, cycle_momentum=False)
        self.epe = EPE()
        self.ter = Ternary()
        self.sobel = SOBEL()
        # self.vgg = VGGPerceptualLoss().to(device)
        if local_rank != -1:
            self.flownet = DDP(self.flownet, device_ids=[
                               local_rank], output_device=local_rank)
            self.contextnet = DDP(self.contextnet, device_ids=[
                                  local_rank], output_device=local_rank)
            self.fusionnet = DDP(self.fusionnet, device_ids=[
                                 local_rank], output_device=local_rank)

    def train(self):
        self.flownet.train()
        self.contextnet.train()
        self.fusionnet.train()

    def eval(self):
        self.flownet.eval()
        self.contextnet.eval()
        self.fusionnet.eval()

    def device(self):
        self.flownet.to(device)
        self.contextnet.to(device)
        self.fusionnet.to(device)

    def load_model(self, path, rank=-1):
        def convert(param):
            if rank == -1:
                return {
                    k.replace("module.", ""): v
                    for k, v in param.items()
                    if "module." in k
                }
            else:
                return param
        if rank <= 0:
            self.flownet.load_state_dict(
                convert(torch.load('{}/flownet.pkl'.format(path), map_location=device)))
            self.contextnet.load_state_dict(
                convert(torch.load('{}/contextnet.pkl'.format(path), map_location=device)))
            self.fusionnet.load_state_dict(
                convert(torch.load('{}/unet.pkl'.format(path), map_location=device)))

    def save_model(self, path, rank):
        if rank == 0:
            torch.save(self.flownet.state_dict(),
                       '{}/flownet.pkl'.format(path))
            torch.save(self.contextnet.state_dict(),
                       '{}/contextnet.pkl'.format(path))
            torch.save(self.fusionnet.state_dict(), '{}/unet.pkl'.format(path))

    def predict(self, imgs, flow_HR, flow_LR_0_1, flow_LR_2_1, training=True):
        img0_HR = imgs[:, :3]
        img1_HR = imgs[:, 3:6]
        img0_LR = imgs[:, 6:9]
        img1_LR = imgs[:, 9:12]
        img2_LR = imgs[:, 12:15]

        c0_HR = self.contextnet(img0_HR, flow_HR[:, :2])
        c1_HR = self.contextnet(img1_HR, flow_HR[:, 2:4])
        
        # Contextnet can be used here for LR images. However IFNet outputs Ft->0 and Ft->1 but Lucas Kanade outputs F0-1 to F1->0. Hence Lucas Kanade 
        # outputs might be needed to converted first.

        flow_HR = F.interpolate(flow_HR, scale_factor=2.0, mode="bilinear",
                                align_corners=False) * 2.0
        flow_LR_0_1 = F.interpolate(flow_LR_0_1, scale_factor=2.0, mode="bilinear",
                                align_corners=False) * 2.0
        flow_LR_2_1 = F.interpolate(flow_LR_2_1, scale_factor=2.0, mode="bilinear",
                                align_corners=False) * 2.0

        refine_output, warped_img0_HR, warped_img1_HR, warped_img0_LR, warped_img2_LR = self.fusionnet(
            img0_HR, img1_HR, img0_LR, img1_LR, img2_LR, flow_HR, flow_LR_0_1, flow_LR_2_1, c0_HR, c1_HR)
        res = torch.sigmoid(refine_output[:, :3]) * 2 - 1
        mask = torch.sigmoid(refine_output[:, 3:4])
        merged_img = warped_img0_HR * mask + warped_img1_HR * (1 - mask)
        pred = merged_img + res
        pred = torch.clamp(pred, 0, 1)
        if training:
            return pred, mask, merged_img, warped_img0_HR, warped_img1_HR, warped_img0_LR, warped_img2_LR, img0_HR, img1_HR, img0_LR, img2_LR
        else:
            return pred

    def inference(self, imgs, scale=None):
        # Concatenation of img0_HR and img1_HR respectively.
        imgs_HR = imgs[:, :6]
        # Concatenation of img0_LR and img2_LR respectively.
        img0_LR = imgs[:, 6:9]
        img1_LR = imgs[:, 9:12]
        img2_LR = imgs[:, 12:15]
        
        flow_HR, _ = self.flownet(imgs_HR)
        
        flow_LR_0_1 = self.calculate_flow(               # Flow from t=0 to t=1 (low sr, high fps video)
            torch.cat((img0_LR, img1_LR), 1))
       
        flow_LR_2_1 = self.calculate_flow(               # Flow from t=2 to t=1 (low sr, high fps video)
            torch.cat((img2_LR, img1_LR), 1))
        
        print(flow_LR_0_1.dtype)
        

        return self.predict(imgs, flow_HR, flow_LR_0_1, flow_LR_2_1, training=False)

    def calculate_flow(self, x):
        x = F.interpolate(x, scale_factor=0.5, mode="bilinear",
                          align_corners=False)
        img0 = x[:, :3].cpu().numpy()
        img1 = x[:, 3:].cpu().numpy()

        num_samples, _, x, y = img0.shape
        flow_batch = np.empty((0, 2, x, y))
        flow_time = []
        for i in range(num_samples):
            img0_single = img0[i, :, :, :].reshape(x, y, 3)
            img1_single = img1[i, :, :, :].reshape(x, y, 3)
            img0_single = cv2.cvtColor(img0_single, cv2.COLOR_BGR2GRAY)
            img1_single = cv2.cvtColor(img1_single, cv2.COLOR_BGR2GRAY)

            start2 = time.time()
            flow_single = cv2.calcOpticalFlowFarneback(img0_single, img1_single, None, pyr_scale=0.2, levels=3,
                                                       winsize=15, iterations=1, poly_n=1, poly_sigma=1.2, flags=0)
            end2 = time.time()
            flow_time.append((end2 - start2) * 1000)
            flow_single = flow_single.reshape(1, 2, x, y)
            flow_batch = np.append(flow_batch, flow_single, axis=0)
        return torch.tensor(flow_batch, dtype=torch.float)

    def update(self, imgs, gt, learning_rate=0, mul=1, training=True):
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate
        if training:
            self.train()
        else:
            self.eval()
            
        imgs_HR = imgs[:, :6]
        imgs_LR = imgs[:, 6:12]
        
        flow_HR, flow_list_HR = self.flownet(imgs_HR)
        flow_LR, flow_list_LR = self.flownet(imgs_LR)
        
        pred, mask, merged_img, warped_img0_HR, warped_img1_HR, warped_img0_HR, warped_img2_LR, img0_HR, img1_HR, img0_LR, img2_LR = self.predict(
            imgs_HR, imgs[:, 6:], flow_HR, flow_LR)
        loss_ter = self.ter(pred, gt).mean()
        if training:
            with torch.no_grad():
                loss_flow = torch.abs(warped_img0_HR - gt).mean()
                loss_mask = torch.abs(
                    merged_img - gt).sum(1, True).float().detach()
                loss_mask = F.interpolate(loss_mask, scale_factor=0.5, mode="bilinear",
                                          align_corners=False).detach()
        else:
            loss_flow = torch.abs(img0_HR - gt).mean()
            loss_mask = 1
        loss_l1 = (((pred - gt) ** 2 + 1e-6) ** 0.5).mean()
        if training:
            self.optimG.zero_grad()
            loss_G = loss_l1 + loss_ter
            # loss_G = self.vgg(pred, gt) + loss_cons + loss_ter
            loss_G.backward()
            self.optimG.step()
        return pred, merged_img, flow_HR, flow_LR, loss_l1, loss_flow, loss_ter, loss_mask


if __name__ == '__main__':
    img0_HR = cv2.imread(
        "/home/hus/Desktop/repos/HSTRNet/images/original_vid/0.png")
    img1_HR = cv2.imread(
        "/home/hus/Desktop/repos/HSTRNet/images/original_vid/1.png")

    img0_LR = cv2.imread(
        "/home/hus/Desktop/repos/HSTRNet/images/2X_vid/0000000.png")
    img1_LR = cv2.imread(
        "/home/hus/Desktop/repos/HSTRNet/images/2X_vid/0000001.png")
    img2_LR = cv2.imread(
        "/home/hus/Desktop/repos/HSTRNet/images/2X_vid/0000002.png")

    padding1_mult = math.floor(img0_HR.shape[0] / 32) + 1
    padding2_mult = math.floor(img0_HR.shape[1] / 32) + 1
    pad1 = (32 * padding1_mult) - img0_HR.shape[0]
    pad2 = (32 * padding2_mult) - img0_HR.shape[1]

    # Padding to meet dimension requirements of the network
    # Done before network call, otherwise slows down the network
    padding1 = nn.ReplicationPad2d((0, pad2, pad1, 0))

    img0_HR = torch.from_numpy(np.transpose(img0_HR, (2, 0, 1))).to(
        device, non_blocking=True).unsqueeze(0).float() / 255.
    img1_HR = torch.from_numpy(np.transpose(img1_HR, (2, 0, 1))).to(
        device, non_blocking=True).unsqueeze(0).float() / 255.
    img0_LR = torch.from_numpy(np.transpose(img0_LR, (2, 0, 1))).to(
        device, non_blocking=True).unsqueeze(0).float() / 255.
    img1_LR = torch.from_numpy(np.transpose(img1_LR, (2, 0, 1))).to(
        device, non_blocking=True).unsqueeze(0).float() / 255.
    img2_LR = torch.from_numpy(np.transpose(img2_LR, (2, 0, 1))).to(
        device, non_blocking=True).unsqueeze(0).float() / 255.

    imgs = torch.cat((img0_HR, img1_HR, img0_LR, img1_LR, img2_LR), 1)
    imgs = padding1(imgs)
    model = Model()
    model.eval()

    result = model.inference(imgs).cpu().detach().numpy()
    result = result[0, :]
    result = np.transpose(result, (1, 2, 0))
    cv2.imshow("win", result)
    cv2.waitKey(10000)

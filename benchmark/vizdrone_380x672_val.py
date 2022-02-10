import argparse
import sys
sys.path.append('.')
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import time
import math
import logging

from model.HSTR_RIFE_v5_scaled import HSTRNet
from model.pytorch_msssim import ssim_matlab
from dataset_vizdrone_380x672 import VizdroneDataset
from dataset import DataLoader

device = "cuda:2"
torch.cuda.set_device(0)

def convert(param):
    return {k.replace("module.", ""): v for k, v in param.items() if "module." in k}

def image_show(result):
    result = result.cpu().detach().numpy()
    result = result[0, :]
    result = np.transpose(result, (1, 2, 0))
    cv2.imshow("win", result)
    cv2.waitKey(2000)

def crop(pred, gt, w, h):
        _, _ , iw, ih = pred.shape
        x = int((iw - w) / 2)
        y = int((ih - h) / 2)
        pred = pred[:, :, x:iw-x, y:ih-y]
        gt = gt[:, :, x:iw-x, y:ih-y]
        return pred, gt

def validate(model, val_data, len_val):
    model.ifnet.eval()
    model.contextnet.eval()
    model.unet.eval()

    psnr_list = []
    ssim_list = []
    total_times = []
    total_rife_time = []
    total_warp_time = []
    total_context_time = []
    total_fusion_time = []

    for valIndex, data in enumerate(val_data):
        
        if(valIndex % 100 == 0):
            print("Validation index " + str(valIndex))
        with torch.no_grad():
            data = data.to(device, non_blocking=True) / 255.0
            hr_img0 = data[:, :3]
            gt = data[:, 6:9]
            hr_img1 = data[:, 3:6]
        
            """image_show(hr_img0)
            image_show(gt)
            image_show(hr_img1)"""

            hr_images = torch.cat((hr_img0, hr_img1), 1)

            lr_img0 = data[:, 9:12]
            lr_img1 = data[:, 12:15]
            lr_img2 = data[:, 15:18]
        
            """image_show(lr_img0)
            image_show(lr_img1)
            image_show(lr_img2)"""
        
            imgs = torch.cat((lr_img0, lr_img1, lr_img2), 1)

            start_time = time.time()
            pred = model.inference(imgs, hr_images)
            total_times.append(time.time() - start_time)
            """total_times.append(time.time()-start_time)
        total_rife_time.append(rife_time)
        total_warp_time.append(warp_time)
        total_context_time.append(context_time)
        total_fusion_time.append(fusion_time)"""

            """image_show(pred)
            cv2.waitKey(2000)
            image_show(gt)
            cv2.waitKey(2000)"""
        
            pred, gt = crop(pred, gt, 380, 672)
            psnr = -10 * math.log10(((gt - pred) * (gt - pred)).mean())
            ssim_ = float(ssim_matlab(pred, gt))

            psnr_list.append(psnr)
            ssim_list.append(ssim_)
        
    
    print("Total time average")
    print(np.mean(total_times))
    """print("RIFE average")
    print(np.mean(total_rife_time))
    print("Warp average")
    print(np.mean(total_warp_time))
    print("ContextNet average")
    print(np.mean(total_context_time))
    print("FusionNet average")
    print(np.mean(total_fusion_time))"""
    return np.mean(psnr_list), np.mean(ssim_list)

if __name__ == "__main__":
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    model = HSTRNet(device)
   
    dataset_val = VizdroneDataset("validation", "/home/mughees/thinclient_drives/VIZDRONE/upsampled/original/", device)
    val_data = DataLoader(dataset_val, batch_size=1, num_workers=0, shuffle=False)

    len_val = dataset_val.__len__()
    
    model.ifnet.load_state_dict(
        convert(torch.load('./model/RIFE_v5/train_log/flownet.pkl', map_location=device)))
    model.ifnet.eval()
    
    model.contextnet.load_state_dict(torch.load('./model_dict_vizdrone/HSTR_contextnet_100.pkl', map_location=device))
    model.contextnet.eval()
    
    model.unet.load_state_dict(torch.load("./model_dict_vizdrone/HSTR_unet_100.pkl", map_location=device))
    model.unet.eval()
    
    print("Validation is starting")
    psnr, ssim_ = validate(model, val_data, len_val)
    print(psnr)
    print(ssim_)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

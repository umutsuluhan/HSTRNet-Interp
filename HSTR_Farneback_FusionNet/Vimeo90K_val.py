import argparse
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

from model.HSTR_Farneback_RIFE_v5 import HSTRNet
from dataset import VimeoDataset, DataLoader
from model.pytorch_msssim import ssim

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device("cuda:3")

def image_show(result):
    result = result.cpu().detach().numpy()
    result = result[0, :]
    result = np.transpose(result, (1, 2, 0))
    cv2.imshow("win", result)
    cv2.waitKey(2000)


def convert(param):
    return {k.replace("module.", ""): v for k, v in param.items() if "module." in k}

def img_to_flownet(img0, img1):
    img0 = img0.unsqueeze(0)
    img1 = img1.unsqueeze(0)
    images = torch.cat((img0, img1), 0)
    images = images.permute(1, 2, 0, 3, 4).to(device)
    return images

def validate(model, val_data, len_val):
    model.ifnet.eval()
    model.contextnet.eval()
    model.unet.eval()

    psnr_list = []
    ssim_list = []
    total_times = []
    
    total_rife_time = []
    total_farneback_time = []
    total_warp_time = []
    total_context_time = []
    total_fusion_time = []

    for valIndex, data in enumerate(val_data):
        
        if(valIndex % 100 == 0):
            print("Validation index " + str(valIndex))

        data = data.to(device, non_blocking=True) / 255.0

        hr_img0 = data[:, :3]
        gt = data[:, 6:9]
        hr_img1 = data[:, 3:6]

        hr_images = torch.cat((hr_img0, hr_img1), 1)

        lr_img0 = data[:, 9:12]
        lr_img1 = data[:, 12:15]
        lr_img2 = data[:, 15:18]

        images_LR_1_0 = img_to_flownet(lr_img1, lr_img0)
        images_LR_1_2 = img_to_flownet(lr_img1, lr_img2)
        lr_images = torch.cat((images_LR_1_0, images_LR_1_2), 1)
        
        imgs = torch.cat((lr_img0, lr_img1, lr_img2), 1)

        start_time = time.time()
        pred,  rife_time, farneback_time, warp_time, context_time, fusion_time = model.inference(imgs, hr_images, lr_images)
        total_times.append(time.time()-start_time)
        total_rife_time.append(rife_time)
        total_farneback_time.append(farneback_time)
        total_warp_time.append(warp_time)
        total_context_time.append(context_time)
        total_fusion_time.append(fusion_time)

        psnr = -10 * math.log10(((gt - pred) * (gt - pred)).mean())
        ssim_ = float(ssim(pred, gt))

        psnr_list.append(psnr)
        ssim_list.append(ssim_)
        
    print("Total time average")
    print(np.mean(total_times))
    print("RIFE average")
    print(np.mean(total_rife_time))
    print("Flownet average")
    print(np.mean(total_farneback_time))
    print("Warp average")
    print(np.mean(total_warp_time))
    print("ContextNet average")
    print(np.mean(total_context_time))
    print("FusionNet average")
    print(np.mean(total_fusion_time))

    return np.mean(psnr_list), np.mean(ssim_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument("--epoch", default=100, type=int)
    parser.add_argument("--data_root", required=True, type=str)
    args = parser.parse_args()

    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    model = HSTRNet(device)

    dataset_val = VimeoDataset("validation", args.data_root, device)
    val_data_last = DataLoader(dataset_val, batch_size=1, num_workers=0, shuffle=False)
    len_val = dataset_val.__len__()

    model.ifnet.load_state_dict(convert(torch.load('/home/mughees/Projects/HSTRNet/HSTR_Flownet2/model/RIFE_v5/train_log/flownet.pkl', map_location=device)))

    model.contextnet.load_state_dict(convert(torch.load('/home/mughees/Projects/HSTRNet/HSTR_Flownet2/model/RIFE_v5/train_log/contextnet.pkl', map_location=device)))
            
    model.unet.load_state_dict(torch.load('/home/mughees/Projects/HSTRNet/HSTR_Flownet2/model_dict/HSTR_unet_99.pkl'))

    psnr,  ssim = validate(model, val_data_last, len_val) 

    print(psnr)
    print(ssim)

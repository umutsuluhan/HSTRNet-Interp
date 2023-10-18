import sys
sys.path.append('.')
import torch
import torch.nn.functional as F
import numpy as np
import random
import time
import argparse

from model.HSTRNet import HSTRNet
from utils.dataset import *
from torch.utils.data import DataLoader
from utils.utils import ssim_matlab
from utils.utils import image_show, convert, calc_psnr
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def validate(model, val_data):
    psnr_list = []
    ssim_list = []
    total_times = []

    for valIndex, data in enumerate(tqdm(val_data)):
        
        data = data.to(device, non_blocking=True) / 255.0

        hr_img0 = data[:, :3]
        gt = data[:, 6:9]
        hr_img1 = data[:, 3:6]

        hr_images = torch.cat((hr_img0, hr_img1), 1)

        lr_img0 = data[:, 9:12]
        lr_img1 = data[:, 12:15]
        lr_img2 = data[:, 15:18]
        
        lr_imgs = torch.cat((lr_img0, lr_img1, lr_img2), 1)

        start_time = time.time()
        pred = model(lr_imgs, hr_images)

        psnr = calc_psnr(gt, pred)
        ssim = float(ssim_matlab(pred, gt))

        psnr_list.append(psnr)
        ssim_list.append(ssim)
        total_times.append(time.time()-start_time)
    return np.mean(psnr_list), np.mean(ssim_list), np.mean(total_times)

if __name__ == "__main__":
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser(description="train")
    parser.add_argument("--dataset_name", default="Vimeo", type=str)
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--model_path", default="./pretrained", type=str)

    args = parser.parse_args()

    print("Evaluating " + str(args.dataset_name) + " located on:" + str(args.dataset_path))

    model = HSTRNet(device)

    if args.dataset_name == "Vimeo":
        dataset_val = VimeoDataset("validation", args.dataset_path, device)
        val_data = DataLoader(dataset_val, batch_size=1, num_workers=0, shuffle=False)
    elif args.dataset_name == "Vizdrone":
        dataset_val = VizdroneDataset("validation", args.dataset_path, device)
        val_data = DataLoader(dataset_val, batch_size=1, num_workers=0, shuffle=False)

    model.load_pretrained(args.model_path)
    model.eval()
    psnr,  ssim, times = validate(model, val_data) 
    print("Average PSNR:" + str(psnr))
    print("Average SSIM:" + str(ssim))
    print("Average time:" + str(times))


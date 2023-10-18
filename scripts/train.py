import sys
sys.path.append('.')
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
import os
import datetime

from model.HSTRNet import HSTRNet
from utils.dataset import VimeoDataset, DataLoader
from utils.utils import ssim_matlab

from utils.utils import image_show, convert, calc_psnr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model):

    logging.info("Device: %s", device)

    #Loading training partition of Vimeo dataset
    dataset_train = VimeoDataset("train", args.data_root, device)
    train_data = DataLoader(dataset_train, batch_size=args.train_batch_size, num_workers=0, drop_last=True, shuffle=True)
    logging.info("Training dataset is loaded")

    #Loading validation partition of Vimeo dataset
    dataset_val = VimeoDataset("validation", args.data_root, device)
    val_data = DataLoader(dataset_val, batch_size=args.val_batch_size, num_workers=0, shuffle=False)
    logging.info("Validation dataset is loaded")

    len_val = dataset_val.__len__()

    # Loss function and optimizer
    L1_lossFn = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    print("Training...")
    logging.info("Training is starting")
    
    # If checkpoint argument true, checkpoint is loaded
    if args.checkpoint == 1:
        model.contextnet.load_state_dict(torch.load(os.path.join(args.checkpoint_path, "contextnet.pkl"), map_location=device))
        model.unet.load_state_dict(torch.load(os.path.join(args.checkpoint_path, "unet.pkl"), map_location=device))
        optimizer.load_state_dict(torch.load(os.path.join(args.checkpoint_path, "optimizer.pkl"), map_location=device))
    
    # Freezing models that are not going to be trained
    model.ifnet = model.freeze(model.ifnet)

    # Below code is a test to check if validation works as expected
    """print("Validation is starting")
    psnr, ssim = validate(model, val_data, len_val, 1)
    print(psnr)
    print(ssim)"""

    start = time.time()
    loss = 0
    psnr_list = []
    ssim_list = []

    if args.checkpoint == 0:
        epoch_start = 0
    else:
        epoch_start = args.checkpoint_start

    for epoch in range(epoch_start, args.epoch):
        model.contextnet.train()
        model.unet.train()
        loss = 0

        print("Epoch: ", epoch)
        logging.info("---------------------------------------------")
        logging.info("Epoch:" + str(epoch))
        logging.info("---------------------------------------------")

        for trainIndex, data in enumerate(train_data):
            model.contextnet.train()
            model.unet.train()            

            if trainIndex % 500 == 0:
                print("Train Index:" + str(trainIndex))
                logging.info("Train Index:" + str(trainIndex))

            # Reading data
            data = data.to(device, non_blocking=True) / 255.0

            hr_img0 = data[:, :3]
            gt = data[:, 6:9]
            hr_img1 = data[:, 3:6]
            
            # Packing HR images together
            hr_images = torch.cat((hr_img0, hr_img1), 1)

            lr_img0 = data[:, 9:12]
            lr_img1 = data[:, 12:15]
            lr_img2 = data[:, 15:18]
            
            # Packing LR images together
            lr_images = torch.cat((lr_img0, lr_img1, lr_img2), 1)
            
            optimizer.zero_grad()

            # Inference
            pred = model(lr_images, hr_images)

            # Loss calculation, backward and optimizer step 
            L1_loss = L1_lossFn(pred, gt)
            L1_loss.backward()
            optimizer.step()
            loss += float(L1_loss.item())
            end = time.time()


            # Validation
            if trainIndex % 500 == 0 and trainIndex != 0:
                print("Validating, Train Index: " + str(trainIndex))
                logging.info("Validating, Train Index: " + str(trainIndex))

                with torch.no_grad():
                    psnr, ssim = validate(model, val_data, len_val, 1)

                    psnr_list.append(psnr)
                    ssim_list.append(ssim)

                    endVal = time.time()

                print(
                    " Loss: %0.6f  TrainExecTime: %0.1f  ValPSNR: %0.4f  ValEvalTime: %0.2f  SSIM: %0.4f "
                    % (loss / trainIndex, end - start, psnr, endVal - end, ssim)
                )
                logging.info(
                    "Train index: "
                    + str(trainIndex)
                    + " Loss: "
                    + str(round(loss / trainIndex, 6))
                    + " TrainExecTime: "
                    + str(round(end - start, 1))
                    + " ValPSNR: "
                    + str(round(psnr, 4))
                    + " ValEvalTime: "
                    + str(round(endVal - end, 2))
                    + " SSIM: "
                    + str(round(ssim, 4))
                )
                start = time.time()

        # Saving model
        if epoch % 5 == 0:
            torch.save(model.contextnet.state_dict(), "model_dict/HSTR_contextnet_" + str(epoch) + ".pkl")
            torch.save(model.unet.state_dict(), "model_dict/HSTR_unet_" + str(epoch) + ".pkl")
            torch.save(optimizer.state_dict(), "model_dict/HSTR_optimizer" + str(epoch) + ".pkl")
        
    # Last epoch validation
    val_data_last = DataLoader(dataset_val, batch_size=1, num_workers=0, shuffle=False)
    psnr, ssim = validate(model, val_data_last, len_val, 1)
    logging.info("------------------------------------------")
    logging.info(
        "Last evaluation --> PSNR:"
        + str(psnr)
        + " SSIM:"
        + str(ssim)
    )


def validate(model, val_data, len_val, batch_size):
    model.eval()
    psnr_list = []
    ssim_list = []

    for valIndex, data in enumerate(val_data):
        if(valIndex % 100 == 0):
            print("Validation index " + str(valIndex))

        with torch.no_grad():
            data = data.to(device, non_blocking=True) / 255.0
    
            hr_img0 = data[:, :3]
            gt = data[:, 6:9]
            hr_img1 = data[:, 3:6]
    
            hr_images = torch.cat((hr_img0, hr_img1), 1)
    
            lr_img0 = data[:, 9:12]
            lr_img1 = data[:, 12:15]
            lr_img2 = data[:, 15:18]
            
            lr_images = torch.cat((lr_img0, lr_img1, lr_img2), 1)
    
            pred = model(lr_images, hr_images)
    
            # PSNR and SSIM calculations
            for i in range(int(pred.shape[0])):
                psnr = calc_psnr(pred[i: i+1,:], gt[i: i+1,:])
                ssim_ = float(ssim_matlab(pred[i: i+1,:], gt[i: i+1,:]))
        
                psnr_list.append(psnr)
                ssim_list.append(ssim_)
    return np.mean(psnr_list), np.mean(ssim_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument("--epoch", default=100, type=int)
    parser.add_argument("--data_root", required=True, type=str)
    parser.add_argument("--train_batch_size", required=True, type=int)
    parser.add_argument("--val_batch_size", required=True, type=int)
    parser.add_argument("--checkpoint", required=True, type=int)
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--checkpoint_start", type=int)
    args = parser.parse_args()
    
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    if not os.path.exists("./logs"):
        os.makedirs("./logs")

    # Logging initialization
    logging.basicConfig(filename="logs/training.log" + str(datetime.datetime.today().strftime('_%d-%m-%H')), filemode="w", format="%(asctime)s - %(message)s", level=logging.INFO,)

    # Instantiating model
    model = HSTRNet(device)
    try:
        # main train function
        train(model)
    except Exception as e:
        logging.exception("Unexpected exception! %s", e)

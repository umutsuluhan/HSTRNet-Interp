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
import datetime

from model.HSTR_RIFE_v5_scaled import HSTRNet
from model.pytorch_msssim import ssim_matlab
from dataset import VizdroneDataset
from dataset import DataLoader

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def image_show(result):
    result = result.cpu().detach().numpy()
    result = result[0, :]
    result = np.transpose(result, (1, 2, 0))
    cv2.imshow("win", result)
    cv2.waitKey(2000)

def convert(param):
    return {k.replace("module.", ""): v for k, v in param.items() if "module." in k}

def train(model):

    logging.info("Device: %s", device)
    
    dataset_train = VizdroneDataset("train", "/home/ortak/mughees/datasets/Vizdrone_format_448x256/", device)
    train_data = DataLoader(
        dataset_train, batch_size=64, num_workers=0, drop_last=True, shuffle=True
    )
    
    logging.info("Training dataset is loaded")

    dataset_val = VizdroneDataset("validation", "/home/ortak/mughees/datasets/Vizdrone_format_448x256/", device)
    val_data = DataLoader(dataset_val, batch_size=16, num_workers=0, shuffle=False)

    logging.info("Validation dataset is loaded")

    len_val = dataset_val.__len__()
    
    L1_lossFn = nn.L1Loss()
    params = model.return_parameters()
    optimizer = optim.Adam(params, lr=0.0001)
    #scheduler = optim.lr_scheduler.MultiStepLR(
    #    optimizer, milestones=[50, 75], gamma=0.1
    #)
    
    print("Training...")
    logging.info("Training is starting")

    model.ifnet.load_state_dict(
        convert(torch.load('./model/RIFE_v5/train_log/flownet.pkl', map_location=device)))
    model.ifnet.eval()
    
    model.contextnet.load_state_dict(torch.load('./model_dict/HSTR_contextnet_200.pkl', map_location=device))
    
    model.unet.load_state_dict(torch.load('./model_dict/HSTR_unet_200.pkl', map_location=device))

    logging.info("Pretrained models are loaded")

    # Freezing models that are not going to be trained
    for k, v in model.ifnet.named_parameters():
        v.requires_grad = False

    logging.info("Required models are freezed")
    
    # Below code is a test to check if validation works as expected.

    print("Validation is starting")
    psnr, ssim = validate(model, val_data, len_val, 1)
    print(psnr)
    print(ssim)
    
    
    start = time.time()

    loss = 0
    psnr_list = []
    ssim_list = []

    for epoch in range(args.epoch):
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

            """if trainIndex % 450 == 0:
                print("Train Index:" + str(trainIndex))
                logging.info("Train Index:" + str(trainIndex))"""

            data = data.to(device, non_blocking=True) / 255.0

            hr_img0 = data[:, :3]
            gt = data[:, 6:9]
            hr_img1 = data[:, 3:6]
            
            hr_images = torch.cat((hr_img0, hr_img1), 1)

            lr_img0 = data[:, 9:12]
            lr_img1 = data[:, 12:15]
            lr_img2 = data[:, 15:18]
            
            lr_images = torch.cat((lr_img0, lr_img1, lr_img2), 1)
            
            """image_show(hr_img0)
            image_show(gt)
            image_show(hr_img1)
            image_show(lr_img0)
            image_show(lr_img1)
            image_show(lr_img2)"""

            optimizer.zero_grad()

            pred = model.inference(lr_images, hr_images)

            L1_loss = L1_lossFn(pred, gt)
            L1_loss.backward()
            optimizer.step()
            loss += float(L1_loss.item())

            end = time.time()
            
            if trainIndex % 400 == 0 and trainIndex != 0 and epoch % 2 == 0:

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

        #scheduler.step()

        #logging.info("---------------------------------------------")
        #logging.info("Learning rate:" + str(scheduler.get_last_lr()))
        #logging.info("---------------------------------------------")

        if epoch % 5 == 0:
            torch.save(model.contextnet.state_dict(), "model_dict_vizdrone/HSTR_contextnet_" + str(epoch) + ".pkl")
            torch.save(model.unet.state_dict(), "model_dict_vizdrone/HSTR_unet_" + str(epoch) + ".pkl")
            torch.save(optimizer.state_dict(), "model_dict_vizdrone/HSTR_optimizer_" + str(epoch) + ".pkl")
        
    val_data_last = DataLoader(dataset_val, batch_size=1, num_workers=0, shuffle=False)

    psnr, ssim = validate(model, val_data_last, len_val, 1)
    logging.info("------------------------------------------")
    logging.info(
        "Last evaluation --> PSNR:"
        + str(psnr)
        + " SSIM:"
        + str(ssim)
    )
    print("Last eval--> PSNR:" + str(psnr) + "  SSIM:"+ str(ssim))
    
            
def validate(model, val_data, len_val, batch_size):
    model.ifnet.eval()
    model.contextnet.eval()
    model.unet.eval()

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
            
            """image_show(hr_img0)
            image_show(gt)
            image_show(hr_img1)
            image_show(lr_img0)
            image_show(lr_img1)
            image_show(lr_img2)"""

            lr_images = torch.cat((lr_img0, lr_img1, lr_img2), 1)

            pred = model.inference(lr_images, hr_images)
        
            for i in range(int(pred.shape[0])):
                psnr = -10 * math.log10(((gt[i: i+1,:] - pred[i: i+1,:]) * (gt[i: i+1,:] - pred[i: i+1,:])).mean())
                ssim_ = float(ssim_matlab(pred[i: i+1,:], gt[i: i+1,:]))
                psnr_list.append(psnr)
                ssim_list.append(ssim_)
    return np.mean(psnr_list) * batch_size, np.mean(ssim_list) * batch_size
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument("--epoch", default=101, type=int)
    args = parser.parse_args()

    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    logging.basicConfig(filename="logs/training_vizdrone.log" + str(datetime.datetime.today().strftime('_%d-%m-%H')), filemode="w", format="%(asctime)s - %(message)s", level=logging.INFO,)

    model = HSTRNet(device)
    #try:
    train(model)
    #except Exception as e:
    #    logging.exception("Unexpected exception! %s", e)


































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
from model.pytorch_msssim import ssim
from dataset_vizdrone import VizdroneDataset
from dataset import DataLoader


device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device("cuda:3")

def image_show(result):
    result = result.cpu().detach().numpy()
    result = result[0, :]
    result = np.transpose(result, (1, 2, 0))
    cv2.imshow("win", result)
    cv2.waitKey(2000)

def normalization(img):
    frame_normed = 255 * (img - img.min()) / (img.max() - img.min())
    frame_normed = np.array(frame_normed, np.int)
    return frame_normed

def convert(param):
    return {k.replace("module.", ""): v for k, v in param.items() if "module." in k}

def img_to_flownet(img0, img1):
    img0 = img0.unsqueeze(0)
    img1 = img1.unsqueeze(0)
    images = torch.cat((img0, img1), 0)
    images = images.permute(1, 2, 0, 3, 4).to(device)
    return images

# dataset = VizdroneDataset("train", "/home/ortak/mughees/datasets/Vizdrone_format_448x256/", device)

# for trainIndex, data in enumerate(dataset):
#     data = data.to(device, non_blocking=True) / 255.0

#     hr_img0 = data[:3]
#     gt = data[6:9]
#     hr_img1 = data[3:6]

#     lr_img0 = data[9:12]
#     lr_img1 = data[12:15]
#     lr_img2 = data[15:18]
    
#     image_show(hr_img0)
#     image_show(gt)
#     image_show(hr_img1)
#     image_show(lr_img0)
#     image_show(lr_img1)
#     image_show(lr_img2)
    
def train(model):

    logging.info("Device: %s", device)
    
    dataset_train = VizdroneDataset("train", "/home/ortak/mughees/datasets/Vizdrone_format_448x256/", device)
    train_data = DataLoader(
        dataset_train, batch_size=96, num_workers=0, drop_last=True, shuffle=True
    )
    
    logging.info("Training dataset is loaded")

    dataset_val = VizdroneDataset("validation", "/home/mughees/thinclient_drives/VIZDRONE/upsampled/original/", device)
    val_data = DataLoader(dataset_val, batch_size=1, num_workers=0, shuffle=False)

    logging.info("Validation dataset is loaded")

    len_val = dataset_val.__len__()
    
    L1_lossFn = nn.L1Loss()
    params = model.return_parameters()
    optimizer = optim.Adam(params, lr=0.001)
    
    print("Training...")
    logging.info("Training is starting")

    model.ifnet.load_state_dict(
        convert(torch.load('./model/RIFE_v5/train_log/flownet.pkl', map_location=device)))
    model.ifnet.eval()
    
    model.contextnet.load_state_dict(
        convert(torch.load('./model/RIFE_v5/train_log/contextnet.pkl', map_location=device)))
    model.contextnet.eval()
    
    model.unet.load_state_dict(torch.load("./model_dict_vizdrone/HSTR_unet_99.pkl"))

    
    logging.info("Pretrained models are loaded")

    # Freezing models that are not going to be trained
    for k, v in model.ifnet.named_parameters():
        v.requires_grad = False
    for k, v in model.contextnet.named_parameters():
        v.requires_grad = False

    logging.info("Required models are freezed")
    
    # Below code is a test to check if validation works as expected.

    print("Validation is starting")
    psnr, ssim = validate(model, val_data, len_val, 1)
    print(psnr)
    print(ssim)
    
    exit()
    start = time.time()

    loss = 0
    psnr_list = []
    ssim_list = []

    for epoch in range(args.epoch):

        model.unet.train()

        loss = 0

        print("Epoch: ", epoch)
        logging.info("---------------------------------------------")
        logging.info("Epoch:" + str(epoch))
        logging.info("---------------------------------------------")

        
        for trainIndex, data in enumerate(train_data):
            model.unet.train()            

            if trainIndex % 100 == 0:
                print("Train Index:" + str(trainIndex))
                logging.info("Train Index:" + str(trainIndex))

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
            
            optimizer.zero_grad()

            pred = model.inference(imgs, hr_images, lr_images)

            L1_loss = L1_lossFn(pred, gt)

            L1_loss.backward()
            optimizer.step()
            loss += float(L1_loss.item())

            end = time.time()
            
            if trainIndex % 100 == 0 and trainIndex != 0:

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
        torch.save(model.unet.state_dict(), "model_dict_vizdrone/HSTR_unet_" + str(epoch) + ".pkl")
        
    val_data_last = DataLoader(dataset_val, batch_size=1, num_workers=0, shuffle=False)

    psnr, vLoss, ssim = validate(model, val_data_last, len_val, 1)
    logging.info("------------------------------------------")
    logging.info(
        "Last evaluation --> PSNR:"
        + str(psnr)
        + " vloss:"
        + str(vLoss)
        + " SSIM:"
        + str(ssim)
    )
    torch.save(model.unet.state_dict(), "{}/final_unet.pkl".format("model_dict_vizdrone"))
            
def validate(model, val_data, len_val, batch_size):
    model.ifnet.eval()
    model.contextnet.eval()
    model.unet.eval()

    psnr_list = []
    ssim_list = []
    total_times = []
    total_rife_time = []
    total_flownet_time = []
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
        
        # image_show(hr_img0)
        # cv2.waitKey(2000)
        # image_show(gt)
        # cv2.waitKey(2000)
        # image_show(hr_img1)
        # cv2.waitKey(2000)

        hr_images = torch.cat((hr_img0, hr_img1), 1)

        lr_img0 = data[:, 9:12]
        lr_img1 = data[:, 12:15]
        lr_img2 = data[:, 15:18]
        
        # image_show(lr_img0)
        # cv2.waitKey(2000)
        # image_show(lr_img1)
        # cv2.waitKey(2000)
        # image_show(lr_img2)
        # cv2.waitKey(2000)
        
        images_LR_1_0 = img_to_flownet(lr_img1, lr_img0)
        images_LR_1_2 = img_to_flownet(lr_img1, lr_img2)
        lr_images = torch.cat((images_LR_1_0, images_LR_1_2), 1)
        
        imgs = torch.cat((lr_img0, lr_img1, lr_img2), 1)

        start_time = time.time()
        pred, rife_time, flownet_time, warp_time, context_time, fusion_time = model.inference(imgs, hr_images, lr_images)
        total_times.append(time.time() - start_time)
        total_times.append(time.time()-start_time)
        total_rife_time.append(rife_time)
        total_flownet_time.append(flownet_time)
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
    print(np.mean(total_flownet_time))
    print("Warp average")
    print(np.mean(total_warp_time))
    print("ContextNet average")
    print(np.mean(total_context_time))
    print("FusionNet average")
    print(np.mean(total_fusion_time))
    return np.mean(psnr_list) * batch_size, np.mean(ssim_list) * batch_size
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument("--epoch", default=100, type=int)
    args = parser.parse_args()

    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    logging.basicConfig(filename="logs/training_vizdrone.log", filemode="w", format="%(asctime)s - %(message)s", level=logging.INFO,)

    model = HSTRNet(device)
    try:
        train(model)
    except Exception as e:
        logging.exception("Unexpected exception! %s", e)


































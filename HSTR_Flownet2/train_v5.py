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

from model.HSTR_Flownet_RIFE_v5 import HSTRNet
from dataset import VimeoDataset, DataLoader
from model.pytorch_msssim import ssim

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device("cuda:1")

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

def train(model):

    logging.info("Device: %s", device)

    dataset_train = VimeoDataset("train", args.data_root, device)
    train_data = DataLoader(
        dataset_train, batch_size=96, num_workers=0, drop_last=True, shuffle=True
    )

    logging.info("Training dataset is loaded")

    dataset_val = VimeoDataset("validation", args.data_root, device)
    val_data = DataLoader(dataset_val, batch_size=8, num_workers=0, shuffle=False)

    logging.info("Validation dataset is loaded")

    len_val = dataset_val.__len__()

    L1_lossFn = nn.L1Loss()
    params = model.return_parameters()
    optimizer = optim.Adam(params, lr=0.001)
    """ scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[100, 150], gamma=0.1
    )  # Look at this, plot learning rate """

    #learning_rates = []

    print("Training...")
    logging.info("Training is starting")

    model.ifnet.load_state_dict(
        convert(torch.load('/home/mughees/Projects/HSTRNet/HSTR_Flownet2/model/RIFE_v5/train_log/flownet.pkl', map_location=device)))
    model.ifnet.eval()
    
    model.contextnet.load_state_dict(
        convert(torch.load('/home/mughees/Projects/HSTRNet/HSTR_Flownet2/model/RIFE_v5/train_log/contextnet.pkl', map_location=device)))
    model.contextnet.eval()

    dict_ = torch.load("/home/mughees/Projects/HSTRNet/HSTR_Flownet2/model/flownet2/train_log/FlowNet2_checkpoint.pth.tar")
    model.flownet2.load_state_dict(dict_["state_dict"])
    model.flownet2.eval()

    # Freezing models that are not going to be trained
    for k, v in model.ifnet.named_parameters():
        v.requires_grad = False
    for k, v in model.contextnet.named_parameters():
        v.requires_grad = False
    for k, v in model.flownet2.named_parameters():
        v.requires_grad = False

    
    # Below code is a test to check if validation works as expected.

    # print("Validation is starting")
    # psnr, ssim = validate(model, val_data, len_val, 1)
    # print(psnr)
    # print(ssim)

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

            if trainIndex % 300 == 0 and trainIndex != 0:

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
        torch.save(model.unet.state_dict(), "model_dict/HSTR_unet_" + str(epoch) + ".pkl")
        
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
    torch.save(model.unet.state_dict(), "{}/final_unet.pkl".format("model_dict"))


def validate(model, val_data, len_val, batch_size):
    model.ifnet.eval()
    model.flownet2.eval()
    model.contextnet.eval()
    model.unet.eval()

    psnr_list = []
    ssim_list = []

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

        pred = model.inference(imgs, hr_images, lr_images)

        psnr = -10 * math.log10(((gt - pred) * (gt - pred)).mean())
        ssim_ = float(ssim(pred, gt))

        psnr_list.append(psnr)
        ssim_list.append(ssim_)

    return np.mean(psnr_list) * batch_size, np.mean(ssim_list) * batch_size


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument("--epoch", default=100, type=int)
    parser.add_argument("--data_root", required=True, type=str)
    parser.add_argument("--fp16", action="store_true", default="False", help="Run model in pseudo-fp16 mode (fp16 storage fp32 math).",)
    parser.add_argument("--rgb_max", type=float, default=255.0)
    args = parser.parse_args()

    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    logging.basicConfig(filename="logs/training.log", filemode="w", format="%(asctime)s - %(message)s", level=logging.INFO,)

    model = HSTRNet(device, args)
    try:
        train(model)
    except Exception as e:
        logging.exception("Unexpected exception! %s", e)

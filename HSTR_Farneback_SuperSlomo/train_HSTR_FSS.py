import argparse
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import time
import datetime
import math
import logging

from model.HSTR_FSS import HSTR_FSS
from dataset import VimeoDataset, DataLoader
from model.pytorch_msssim import ssim, ssim_matlab

device = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")


def img_to_jpg(img):
    test = img[:1, :]
    test = test[0, :]
    test = test.cpu().detach().numpy()
    test = np.transpose(test, (1, 2, 0))
    test = 255 * (test - test.min()) / (test.max() - test.min())
    test = np.array(test, np.int)
    return test


def convert(param):
    return {k.replace("module.", ""): v for k, v in param.items() if "module." in k}


def train(model):

    logging.info("Device: %s", device)

    dataset_train = VimeoDataset("train", args.data_root, device)
    train_data = DataLoader(
        dataset_train, batch_size=12, num_workers=0, drop_last=True, shuffle=True
    )

    logging.info("Training dataset is loaded")

    dataset_val = VimeoDataset("validation", args.data_root, device)
    val_data = DataLoader(dataset_val, batch_size=6, num_workers=0, shuffle=False)

    logging.info("Validation dataset is loaded")

    len_val = dataset_val.__len__()

    L1_lossFn = nn.L1Loss()
    params = model.return_parameters()
    optimizer = optim.Adam(params, lr=0.001)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[100, 150], gamma=0.1
    )

    print("Training...")
    logging.info("Training is starting")

    pretrained_dict = convert(
        torch.load(
            "/home/hus/Desktop/repos/HSTRNet/HSTR_Farneback_SuperSlomo/trained_model/train_log/flownet.pkl",
            map_location=device,
        )
    )

    # Below code is to remove unet parameters from the model
    pretrained_dict = {
        k: v for (k, v) in pretrained_dict.items() if k[0] == "b" or k[0] == "c"
    }

    # Below code can be used to explore the new trained dict contents.  k--> name, v --> value
    # for k, v in pretrained_dict.items():
    #     print(k)
    #     print(torch.mean(v))

    # Strict is set to false to ignore unet parameters is missing error
    model.flownet.load_state_dict(pretrained_dict, strict=False)

    # Freezing flownet(IFNet) model to not train it.
    for k, v in model.flownet.named_parameters():
        v.requires_grad = False

    print("Validation is starting")

    # Below code is a test to check if validation works as expected.

    psnr, ssim = validate(model, val_data, len_val)
    # print(psnr)
    # print(ssim)
    # print(ssim_mat)

    start = time.time()

    loss = 0
    psnr_list = []
    ssim_list = []

    for epoch in range(args.epoch):

        model.unet.train()

        print("Epoch: ", epoch)
        logging.info("---------------------------------------------")
        logging.info("Epoch:" + str(epoch))
        logging.info("---------------------------------------------")

        for trainIndex, data in enumerate(train_data):

            if trainIndex % 100 == 0:
                print("Train Index:" + str(trainIndex))
                logging.info("Train Index:" + str(trainIndex))

            data = data.to(device, non_blocking=True) / 255.0

            img0_HR = data[:, :3]
            gt = data[:, 6:9]
            img1_HR = data[:, 3:6]

            img0_LR = data[:, 9:12]
            img1_LR = data[:, 12:15]
            img2_LR = data[:, 15:18]

            imgs = torch.cat((img0_HR, img1_HR, img0_LR, img1_LR, img2_LR), 1)

            optimizer.zero_grad()

            pred = model.inference(imgs, [])

            L1_loss = L1_lossFn(pred, gt)

            L1_loss.backward()
            optimizer.step()
            loss += float(L1_loss.item())

            scheduler.step()

            end = time.time()

            if trainIndex % 1000 == 0 and trainIndex != 0:

                print("Validating, Train Index: " + str(trainIndex))
                logging.info("Validating, Train Index: " + str(trainIndex))

                with torch.no_grad():
                    psnr, ssim = validate(model, val_data, len_val, 6)

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

            model.unet.train()
        ssim = 0

        torch.save(model.unet.state_dict(), "model_dict/HSTR" + str(epoch) + ".pkl")

    model.unet.eval()

    val_data_last = DataLoader(dataset_val, batch_size=1, num_workers=0, shuffle=False)

    psnr, vLoss, ssim = validate(model, val_data_last, len_val)
    logging.info("------------------------------------------")
    logging.info(
        "Last evaluation --> PSNR:"
        + str(psnr)
        + " vloss:"
        + str(vLoss)
        + " SSIM:"
        + str(ssim)
    )
    torch.save(model.unet.state_dict(), "{}/unet.pkl".format("model_dict"))


def validate(model, val_data, len_val, batch_size=1):
    model.unet.eval()

    psnr_list = []
    ssim_list = []
    ssim_mat_list = []

    for trainIndex, data in enumerate(val_data):

        data = data.to(device, non_blocking=True) / 255.0

        img0_HR = data[:, :3]
        gt = data[:, 6:9]
        img1_HR = data[:, 3:6]

        img0_LR = data[:, 9:12]
        img1_LR = data[:, 12:15]
        img2_LR = data[:, 15:18]

        imgs = torch.cat((img0_HR, img1_HR, img0_LR, img1_LR, img2_LR), 1)
        pred = model.inference(imgs, [])

        psnr = -10 * math.log10(((gt - pred) * (gt - pred)).mean())
        ssim_ = float(ssim(pred, gt))
        
        psnr_list.append(psnr)
        ssim_list.append(ssim_)

    return np.mean(psnr_list) * batch_size, np.mean(ssim_list) * batch_size


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument("--epoch", default=25, type=int)
    parser.add_argument("--data_root", required=True, type=str)
    args = parser.parse_args()

    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    logging.basicConfig(
        filename="logs/training.log",
        filemode="w",
        format="%(asctime)s - %(message)s",
        level=logging.INFO,
    )

    model = HSTR_FSS(device)
    try:
        train(model)
    except Exception as e:
        logging.exception("Unexpected exception! %s", e)

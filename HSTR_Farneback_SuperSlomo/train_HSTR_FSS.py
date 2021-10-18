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
from model.pytorch_msssim import ssim

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def img_to_jpg(img):
    test = img[:1,:]
    test = test[0,:]
    test = test.cpu().detach().numpy()
    test = np.transpose(test, (1, 2, 0))
    test = 255 * (test - test.min()) / (test.max() - test.min())
    test = np.array(test, np.int)
    return test
    

def train(model):
    
    logging.info("Device: %s", device)
    logging.info("Batch size: " + str(args.batch_size))

    dataset_train = VimeoDataset('train', args.data_root)
    train_data = DataLoader(
        dataset_train, batch_size=12, num_workers=0, drop_last=True, shuffle=True)

    logging.info("Training dataset is loaded")

    dataset_val = VimeoDataset('validation', args.data_root)
    val_data = DataLoader(
        dataset_val, batch_size=12,  num_workers=0, shuffle=False)

    logging.info("Validation dataset is loaded")

    len_val = dataset_val.__len__()

    L1_lossFn = nn.L1Loss()
    params = model.return_parameters()
    optimizer = optim.Adam(params, lr=0.0001)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[100, 150], gamma=0.1)

    dict1 = {'loss': [], 'valLoss': [], 'valPSNR': [], 'epoch': -1}

    cLoss = dict1['loss']
    valLoss = dict1['valLoss']
    valPSNR = dict1['valPSNR']

    print('Training...')
    logging.info("Training is starting")

    start = time.time()

    for epoch in range(args.epoch):

        model.unet.train()

        print("Epoch: ", epoch)
        logging.info("---------------------------------------------")
        logging.info("Epoch:" + str(epoch))
        logging.info("---------------------------------------------")

        cLoss.append([])
        valLoss.append([])
        valPSNR.append([])
        iloss = 0

        for trainIndex, data in enumerate(train_data):

            data = data.to(device, non_blocking=True) / 255.
            
            img0_HR = data[:, :3]
            """test = img_to_jpg(img0_HR)
            cv2.imwrite("logs/train/hr0.jpg", test)"""
            
            gt = data[:, 6:9]
            """test = img_to_jpg(gt)
            cv2.imwrite("logs/train/hr1.jpg", test)"""

            img1_HR = data[:, 3:6]
            """test = img_to_jpg(img1_HR)
            cv2.imwrite("logs/train/hr2.jpg", test)"""

            img0_LR = data[:, 9:12]
            """test = img_to_jpg(img0_LR)
            cv2.imwrite("logs/train/lr0.jpg", test)"""

            img1_LR = data[:, 12:15]
            """test = img_to_jpg(img1_LR)
            cv2.imwrite("logs/train/lr1.jpg", test)"""
            
            img2_LR = data[:, 15:18]
            """test = img_to_jpg(img2_LR)
            cv2.imwrite("logs/train/lr2.jpg", test)"""

            imgs = torch.cat((img0_HR, img1_HR, img0_LR, img1_LR, img2_LR), 1)

            optimizer.zero_grad()

            pred = model.inference(imgs, [])

            L1_loss = L1_lossFn(pred, gt)

            L1_loss.backward()
            optimizer.step()
            iloss += float(L1_loss.item())

            scheduler.step()

            end = time.time()

            if(trainIndex % 500 == 0 and trainIndex != 0):

                print("Validating, Train Index: " + str(trainIndex))
                logging.info("Validating, Train Index: " + str(trainIndex))

                with torch.no_grad():
                    psnr, vLoss, ssim = validate(
                        model, val_data, len_val)

                    valPSNR[epoch].append(psnr)
                    valLoss[epoch].append(vLoss)

                    endVal = time.time()

                print(" Loss: %0.6f  TrainExecTime: %0.1f  ValLoss:%0.6f  ValPSNR: %0.4f  ValEvalTime: %0.2f  SSIM: %0.4f " % (
                    iloss / trainIndex, end - start, vLoss, psnr, endVal - end, ssim))
                logging.info("Train index: " + str(trainIndex) + " Loss: " + str(round(iloss / trainIndex, 6)) +
                             " TrainExecTime: " + str(round(end - start, 1)) + " ValLoss: " + str(round(vLoss.item(), 6)) +
                             " ValPSNR: " + str(round(psnr, 4)) + " ValEvalTime: " + str(round(endVal - end, 2)) +
                             " SSIM: " + str(round(ssim, 4)))
                start = time.time()

            model.unet.train()

        if ((epoch % 10) == 0):
            dict1 = {
                'Detail': "End to end Super SloMo.",
                'epoch': epoch,
                'timestamp': datetime.datetime.now(),
                'trainBatchSz': 4,
                'loss': cLoss,
                'valLoss': valLoss,
                'valPSNR': valPSNR,
                'SSIM': ssim,
                'state_dict_model': model.unet.state_dict(),
            }
            torch.save(dict1, "model_dict" + "/HSTR_" +
                       str(epoch) + ".ckpt")

    torch.save(model.unet.state_dict(), '{}/unet.pkl'.format("model_dict"))


def validate(model, val_data, len_val):
    model.unet.eval()

    val_loss = 0
    psnr = 0
    out_ssim = 0

    L1_lossFn = nn.L1Loss()
    MSE_LossFn = nn.MSELoss()

    for trainIndex, data in enumerate(val_data):

        data = data.to(device, non_blocking=True) / 255.
            
        img0_HR = data[:, :3]
        """test = img_to_jpg(img0_HR)
        cv2.imwrite("logs/val/hr0.jpg", test)"""
            
        gt = data[:, 6:9]
        """test = img_to_jpg(gt)
        cv2.imwrite("logs/val/hr1.jpg", test)"""

        img1_HR = data[:, 3:6]
        """test = img_to_jpg(img1_HR)
        cv2.imwrite("logs/val/hr2.jpg", test)"""

        img0_LR = data[:, 9:12]
        """test = img_to_jpg(img0_LR)
        cv2.imwrite("logs/val/lr0.jpg", test)"""

        img1_LR = data[:, 12:15]
        """test = img_to_jpg(img1_LR)
        cv2.imwrite("logs/val/lr1.jpg", test)"""
            
        img2_LR = data[:, 15:18]
        """test = img_to_jpg(img2_LR)
        cv2.imwrite("logs/val/lr2.jpg", test)"""

        imgs = torch.cat((img0_HR, img1_HR, img0_LR, img1_LR, img2_LR), 1)

        pred = model.inference(imgs, [])

        L1_loss = L1_lossFn(pred, gt)

        val_loss += float(L1_loss)

        MSE_loss = MSE_LossFn(pred, gt)
        psnr += float((10 * math.log10(1 / MSE_loss.item())))
        out_ssim += float(ssim(pred, gt))

    return (psnr / len_val), (val_loss / len_val), (out_ssim / len_val)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--epoch', default=300, type=int)
    parser.add_argument('--batch_size', default=6, type=int,
                        help='minibatch size')  # 4 * 12 = 48
    parser.add_argument('--data_root', required=True, type=str)
    args = parser.parse_args()

    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    logging.basicConfig(filename='logs/training.log', filemode='w',
                        format='%(asctime)s - %(message)s', level=logging.INFO)

    model = HSTR_FSS()
    try:
        train(model)
    except Exception as e:
        logging.exception("Unexpected exception! %s", e)

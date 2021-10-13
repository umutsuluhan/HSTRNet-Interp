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

from model.FastRIFE_Super_Slomo.HSTR_FSS import HSTR_FSS
from dataset import VimeoDataset, DataLoader
from model.pytorch_msssim import ssim

device =  torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def train(model, data_root_p):
    data_root = data_root_p   # MODIFY IN SERVER
    logging.basicConfig(filename='logs/training.log', filemode='w',
                        format='%(asctime)s - %(message)s', level=logging.INFO)

    logging.info("Device: %s", device)
    logging.info("Batch size: " + str(args.batch_size))

    dataset_HR = VimeoDataset('train',data_root, HR = True)
    train_data_HR = DataLoader(
        dataset_HR, batch_size=args.batch_size, num_workers=0, drop_last=True)
    args.step_per_epoch = train_data_HR.__len__()

    logging.info("Training dataset of HR videos are loaded")

    dataset_LR = VimeoDataset('train',data_root, HR=False)
    train_data_LR = DataLoader(
        dataset_LR, batch_size=args.batch_size, num_workers=0, drop_last=True)
    args.step_per_epoch = train_data_LR.__len__()

    logging.info("Training dataset of LR videos are loaded")

    dataset_val_HR = VimeoDataset('validation', data_root, HR=True)
    val_data_HR = DataLoader(
        dataset_val_HR, batch_size=args.batch_size,  num_workers=0)

    logging.info("Validation dataset of HR videos are loaded")

    dataset_val_LR = VimeoDataset('validation', data_root, HR=False)
    val_data_LR = DataLoader(
        dataset_val_LR, batch_size=args.batch_size, num_workers=0)

    logging.info("Validation dataset of LR videos are loaded")

    len_val = dataset_val_HR.__len__()

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
    checkpoint_counter = 0

    for epoch in range(args.epoch):

        print("Epoch: ", epoch)
        logging.info("---------------------------------------------")
        logging.info("Epoch:" + str(epoch))
        logging.info("---------------------------------------------")

        cLoss.append([])
        valLoss.append([])
        valPSNR.append([])
        iloss = 0

        for trainIndex, (data_HR, data_LR) in enumerate(zip(train_data_HR, train_data_LR)):

            data_HR = data_HR.to(device, non_blocking=True) / 255.
            data_LR = data_LR.to(device, non_blocking=True) / 255.

            img0_HR = data_HR[:, :3]
            gt = data_HR[:, 6:9]
            img1_HR = data_HR[:, 3:6]

            img0_LR = data_LR[:, :3]
            img0_LR = F.interpolate(img0_LR, scale_factor=4, mode='bicubic') 

            img1_LR = data_LR[:, 6:9]
            img1_LR = F.interpolate(img1_LR, scale_factor=4, mode='bicubic')

            img2_LR = data_LR[:, 3:6]
            img2_LR = F.interpolate(img2_LR, scale_factor=4, mode='bicubic')

            imgs = torch.cat((img0_HR, img1_HR, img0_LR, img1_LR, img2_LR), 1)

            optimizer.zero_grad()

            pred, g_I0_F_t_0, g_I1_F_t_1, warped_lr_img0, lr_img0, warped_lr_img2, lr_img2 = model.inference(
                imgs, [], training=True)

            L1_loss = L1_lossFn(pred, gt)

            warp_loss = L1_lossFn(g_I0_F_t_0, gt) + L1_lossFn(g_I1_F_t_1, gt) + L1_lossFn(
                warped_lr_img0, lr_img0) + L1_lossFn(warped_lr_img2, lr_img2)

            loss = L1_loss * 0.8 + warp_loss * 0.4
    
            del L1_loss, warp_loss

            loss.backward()
            optimizer.step()
            iloss += float(loss.item())

            scheduler.step()

            end = time.time()

        
            if(trainIndex % 500 == 0):

                print("Validating, Train Index: " + str(trainIndex))
                logging.info("Validating, Train Index: " + str(trainIndex))
    
                with torch.no_grad():
                    psnr, vLoss, ssim = validate(
                        model, val_data_HR, val_data_LR, len_val)

                    valPSNR[epoch].append(psnr)
                    valLoss[epoch].append(vLoss)

                    endVal = time.time()

                print(" Loss: %0.6f  TrainExecTime: %0.1f  ValLoss:%0.6f  ValPSNR: %0.4f  ValEvalTime: %0.2f  SSIM: %0.4f " % (
                    iloss / 100, end - start, vLoss, psnr, endVal - end, ssim))
                logging.info("Train index: " + str(trainIndex) + " Loss: " + str(round(iloss / 100, 6)) +
                             " TrainExecTime: " + str(round(end - start, 1)) + " ValLoss: " + str(round(vLoss.item(), 6)) +
                             " ValPSNR: " + str(round(psnr, 4)) + " ValEvalTime: " + str(round(endVal - end, 2)) +
                             " SSIM: " + str(round(ssim, 4)))
                start = time.time()

        if ((epoch % checkpoint_counter) == 0):
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
                       str(checkpoint_counter) + ".ckpt")
            checkpoint_counter += 1

    torch.save(model.unet.state_dict(), '{}/unet.pkl'.format("model_dict"))

def validate(model, val_data_HR, val_data_LR, len_val):
    model.unet.eval()

    val_loss = 0
    psnr = 0
    out_ssim = 0

    L1_lossFn = nn.L1Loss()
    MSE_LossFn = nn.MSELoss()

    for i, (data_HR, data_LR) in enumerate(zip(val_data_HR, val_data_LR)):
        data_HR = data_HR.to(device, non_blocking=True) / 255.
        data_LR = data_LR.to(device, non_blocking=True) / 255.

        img0_HR = data_HR[:, :3]
        gt = data_HR[:, 6:9]
        img1_HR = data_HR[:, 3:6]

        img0_LR = data_LR[:, :3]
        img0_LR = F.interpolate(img0_LR, scale_factor=4, mode='bicubic')

        img1_LR = data_LR[:, 6:9]
        img1_LR = F.interpolate(img1_LR, scale_factor=4, mode='bicubic')

        img2_LR = data_LR[:, 3:6]
        img2_LR = F.interpolate(img2_LR, scale_factor=4, mode='bicubic')

        imgs = torch.cat((img0_HR, img1_HR, img0_LR, img1_LR, img2_LR), 1)

        pred, g_I0_F_t_0, g_I1_F_t_1, warped_lr_img0, lr_img0, warped_lr_img2, lr_img2 = model.inference(
            imgs, [], training=True)

        L1_loss = L1_lossFn(pred, gt)

        warp_loss = L1_lossFn(g_I0_F_t_0, gt) + L1_lossFn(g_I1_F_t_1, gt) + L1_lossFn(
            warped_lr_img0, lr_img0) + L1_lossFn(warped_lr_img2, lr_img2)

        loss = L1_loss * 0.8 + warp_loss * 0.4

        del L1_loss, warp_loss

        val_loss += float(loss)

        MSE_loss = MSE_LossFn(pred, gt)
        psnr += float((10 * math.log10(1 / MSE_loss.item())))
        out_ssim += float(ssim(pred, gt))
    
    
    return (psnr / len_val), (loss / len_val), (out_ssim / len_val)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--epoch', default=300, type=int)
    parser.add_argument('--batch_size', default=12, type=int,
                        help='minibatch size')  # 4 * 12 = 48
    parser.add_argument('--data_root', required=True, type=str)
    args = parser.parse_args()

    # ASK IF NEEDED
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    model = HSTR_FSS()
    try:
        train(model, args.data_root)
    except Exception as e:
        logging.exception("Unexpected exception! %s", e)

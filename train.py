import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import time
import math

from model.FastRIFE_Super_Slomo.HSTR_FSS import HSTR_FSS
from torch.utils.tensorboard import SummaryWriter
from dataset import VimeoDataset, DataLoader

device = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, data_root_p):
    data_root = data_root_p   # MODIFY IN SERVER
    log_path = 'logs'
    writer = SummaryWriter(log_path + '/train')
    writer_val = SummaryWriter(log_path + '/validate')
    step = 0

    dataset_HR = VimeoDataset('train', data_root)
    train_data_HR = DataLoader(
        dataset_HR, batch_size=args.batch_size, num_workers=8, pin_memory=True, drop_last=True)
    args.step_per_epoch = train_data_HR.__len__()

    dataset_LR = VimeoDataset('train', data_root)
    train_data_LR = DataLoader(
        dataset_LR, batch_size=args.batch_size, num_workers=8, pin_memory=True, drop_last=True)
    args.step_per_epoch = train_data_LR.__len__()

    dataset_val_HR = VimeoDataset('validation', data_root)
    val_data_HR = DataLoader(
        dataset_val_HR, batch_size=12, pin_memory=True, num_workers=8)

    dataset_val_LR = VimeoDataset('validation', data_root)
    val_data_LR = DataLoader(
        dataset_val_LR, batch_size=12, pin_memory=True, num_workers=8)

    validate(model, val_data_HR, val_data_LR, writer_val)
    # model.save_model(log_path)

    L1_lossFn = nn.L1Loss()
    params = model.return_parameters()
    optimizer = optim.Adam(params, lr=0.0001)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[100, 150], gamma=0.1)
    loss_list = list()

    print('training...')

    for epoch in range(args.epoch):
        
        scheduler.step()
        
        for i, (data_HR, data_LR) in enumerate(zip(train_data_HR, train_data_LR)):

            data_HR = data_HR.to(device, non_blocking=True) / 255.
            data_LR = data_LR.to(device, non_blocking=True) / 255.

            img0_HR = data_HR[:, :3]
            gt = data_HR[:, 6:9]
            img1_HR = data_HR[:, 3:6]

            img0_LR = data_LR[:, :3]
            img1_LR = data_LR[:, 6:9]
            img2_LR = data_LR[:, 3:6]

            imgs = torch.cat((img0_HR, img1_HR, img0_LR, img1_LR, img2_LR), 1)

            optimizer.zero_grad()

            pred, g_I0_F_t_0, g_I1_F_t_1, warped_lr_img0, lr_img0, warped_lr_img2, lr_img2 = model.inference(
                imgs, [], training=True)

            L1_loss = L1_lossFn(pred, gt)

            warp_loss = L1_lossFn(g_I0_F_t_0, gt) + L1_lossFn(g_I1_F_t_1, gt) + L1_lossFn(
                warped_lr_img0, lr_img0) + L1_lossFn(warped_lr_img2, lr_img2)

            loss = L1_loss * 0.8 + warp_loss * 0.4

            loss.backward()
            optimizer.step()
            loss_list.append(loss)

            if i % 100 == 1:
                writer.add_scalar("loss", loss, step)
                writer.add_scalar('L1_loss', L1_loss, step)
                writer.add_scalar('warp_loss', warp_loss, step)


def validate(model, val_data_HR, val_data_LR, writer_val):
    val_loss = []
    psnr_list = []

    L1_lossFn = nn.L1Loss()
    MSE_LossFn = nn.MSELoss()

    for i, (data_HR, data_LR) in enumerate(zip(val_data_HR, val_data_LR)):
        data_HR = data_HR.to(device, non_blocking=True) / 255.
        data_LR = data_LR.to(device, non_blocking=True) / 255.

        img0_HR = data_HR[:, :3]
        gt = data_HR[:, 6:9]
        img1_HR = data_HR[:, 3:6]

        img0_LR = data_LR[:, :3]
        img1_LR = data_LR[:, 6:9]
        img2_LR = data_LR[:, 3:6]

        imgs = torch.cat((img0_HR, img1_HR, img0_LR, img1_LR, img2_LR), 1)

        pred, g_I0_F_t_0, g_I1_F_t_1, warped_lr_img0, lr_img0, warped_lr_img2, lr_img2 = model.inference(
            imgs, [], training=True)

        L1_loss = L1_lossFn(pred, gt)

        warp_loss = L1_lossFn(g_I0_F_t_0, gt) + L1_lossFn(g_I1_F_t_1, gt) + L1_lossFn(
            warped_lr_img0, lr_img0) + L1_lossFn(warped_lr_img2, lr_img2)

        loss = L1_loss * 0.8 + warp_loss * 0.4

        val_loss.append(loss)

        MSE_loss = MSE_LossFn(pred, gt)
        psnr = (10 * math.log10(1 / MSE_loss.item()))
        psnr_list.append(psnr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--epoch', default=300, type=int)
    parser.add_argument('--batch_size', default=12, type=int,
                        help='minibatch size')  # 4 * 12 = 48
    parser.add_argument('--data_root', required=True, type=str)
    args = parser.parse_args()

    # ASK IF NEEDED
    # seed = 1234
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = True

    model = HSTR_FSS()
    train(model, args.data_root)

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import time

from model.FastRIFE_Super_Slomo.HSTR_FSS import HSTR_FSS
from torch.utils.tensorboard import SummaryWriter
from dataset import *

device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")


# def validate(model, val_data_HR, val_data_LR, writer_val):
    

def train(model, data_root_p):
    data_root = data_root_p   # MODIFY IN SERVER
    log_path = 'logs'
    writer = SummaryWriter(log_path + '/train')
    writer_val = SummaryWriter(log_path + '/validate')
    step = 0
    nr_eval = 0
    
    dataset_HR = VimeoDataset('train', data_root)
    train_data_HR = DataLoader(dataset_HR, batch_size=args.batch_size, num_workers=8, pin_memory=True, drop_last=True)
    args.step_per_epoch = train_data_HR.__len__()
    
    dataset_LR = VimeoDataset('train', data_root)
    train_data_LR = DataLoader(dataset_LR, batch_size=args.batch_size, num_workers=8, pin_memory=True, drop_last=True)
    args.step_per_epoch = train_data_LR.__len__()
    
    dataset_val_HR = VimeoDataset('validation', data_root)
    val_data_HR = DataLoader(dataset_val_HR, batch_size=12, pin_memory=True, num_workers=8)
    
    dataset_val_LR = VimeoDataset('validation', data_root)
    val_data_LR = DataLoader(dataset_val_LR, batch_size=12, pin_memory=True, num_workers=8)
    
    
    #evaluate(model, val_data, nr_eval, writer_val)
    #model.save_model(log_path)
    
    L1_lossFn = nn.L1Loss()
    params = model.return_parameters()
    optimizer = optim.Adam(params, lr=0.0001)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
    loss_list = list()
    
    print('training...')
    
   
    
    for epoch in range(args.epoch):
        for i , (data_HR, data_LR) in enumerate(zip(train_data_HR, train_data_LR)):
            print(i)
            time_stamp = time.time()
            # data_HR = data_HR.to(device, non_blocking=True) / 255.
            # data_LR = data_LR.to(device, non_blocking=True) / 255.
            data_HR = data_HR / 255.
            data_LR = data_LR / 255.
            # data_gpu_HR = data_HR
            # data_gpu = data_gpu.to(device, non_blocking=True) / 255.
            img0_HR = data_HR[:, :3]
            gt = data_HR[:, 6:9]
            img1_HR = data_HR[:, 3:6]
            
            data_gpu_LR = data_LR
            img0_LR = data_LR[:, :3]
            img1_LR = data_LR[:, 6:9]
            img2_LR = data_LR[:, 3:6]
            
            imgs = torch.cat((img0_HR, img1_HR, img0_LR, img1_LR, img2_LR), 1)
            
            optimizer.zero_grad()
            
            pred, g_I0_F_t_0, g_I1_F_t_1, warped_lr_img0, lr_img0, warped_lr_img2, lr_img2 = model.inference(imgs, [], training=True)
            # cv2.imshow("win", pred)
            # cv2.waitKey(2000)
           
            
            L1_loss = L1_lossFn(pred, gt)
            warp_loss = L1_lossFn(g_I0_F_t_0, gt) + L1_lossFn(g_I1_F_t_1, gt) + L1_lossFn(warped_lr_img0, lr_img0) + L1_lossFn(warped_lr_img2, lr_img2)
            
            loss = L1_loss * 0.8 + warp_loss * 0.4
            
            loss.backward()
            optimizer.step()
            loss_list.append(loss)
            
            if i % 100 == 1:
                writer.add_scalar("loss", loss, step)
                writer.add_scalar('L1_loss', L1_loss, step)
                writer.add_scalar('warp_loss', warp_loss, step)

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='hey')
    parser.add_argument('--epoch', default=300, type=int)
    parser.add_argument('--batch_size', default=12, type=int, help='minibatch size') # 4 * 12 = 48
    parser.add_argument('--data_root', required=True, type=str)
    args = parser.parse_args()
    device = torch.device("cuda")
    
    # ASK IF NEEDED
    # seed = 1234
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = True
   
    model = HSTR_FSS()
    train(model, args.data_root)

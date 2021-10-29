#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 11:25:00 2020

@author: hmahmad
"""

import numpy as np
import matplotlib.pyplot as plt
from pytorch_msssim import ssim#, ms_ssim
from math import log10
import os


def show_time(now):
    s = str(now.year) + '/' + str(now.month) + '/' + str(now.day) + ' ' \
        + '%02d' % now.hour + ':' + '%02d' % now.minute + ':' + '%02d' % now.second
    return s

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
   
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def show_figure(gt, out, frame, model_name_):
    #varn = var.numpy()
    gt = (np.transpose(gt.numpy(),[1,2,0])* 255).astype(np.uint8)
    out = np.clip((np.transpose(out.numpy(),[1,2,0])* 255).astype(np.uint8),0,255)
    video_folder,frame_label = frame.split('/')
    frame = video_folder + '_' + frame_label
    plt.subplot(121)
    plt.imshow(gt)
    plt.title('GT: {}'.format(frame))
    #plt.imsave('./visual_results/Base_model/{}_GT.png'.format(frame),gt)
    plt.subplot(122)
    plt.imshow(out)
    mkdir('./visual_results/'+ model_name_)
    plt.imsave('./visual_results/{}/{}_OUTPUT.png'.format(model_name_,frame),out)
    plt.title('OUTPUT: {}'.format(frame))
    plt.show()
    
def rmse(predictions, targets):
    mse = ((predictions - targets) ** 2).mean()
    rmse_ = np.sqrt(mse)
    return {'MSE': mse,
            "RMSE": rmse_}

def PSNR_function(rmse, max_pixel = 1.0): 
    psnr = 20 * log10(max_pixel / rmse) 
    return psnr 


def PSNR_(original, compressed, max_pixel = 1.0): 
    mse = ((original - compressed) ** 2).mean() 
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                  # Therefore PSNR have no importance. 
        return 100
    psnr = 20 * log10(max_pixel / np.sqrt(mse)) 
    return psnr 



# def show_figure(gt, out, frame):
#     #varn = var.numpy()
#     gt = np.clip((np.transpose(gt.numpy(),[1,2,0])* 255).astype(np.uint8), 0,255)
#     out = np.clip((np.transpose(out.numpy(),[1,2,0])* 255).astype(np.uint8),0.255)
#     video_folder,frame_label = frame.split('/')
#     frame = video_folder + '_' + frame_label
#     plt.subplot(121)
#     plt.imshow((np.transpose(gt.numpy(),[1,2,0])* 255).astype(np.uint8))
#     plt.title('GT: {}'.format(frame))
#     plt.imsave('./visual_results/HR_included/GT{}.png'.format(frame),(np.transpose(gt.numpy(),[1,2,0])* 255).astype(np.uint8))
#     plt.subplot(122)
#     plt.imshow((np.transpose(out.numpy(),[1,2,0])* 255).astype(np.uint8))
#     plt.imsave('./visual_results/HR_included/OUTPUT{}.png'.format(frame),(np.transpose(out.numpy(),[1,2,0])* 255).astype(np.uint8))
#     plt.title('OUTPUT: {}'.format(frame))
#     plt.show()

    

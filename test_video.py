import cv2
import math
import torch
import argparse

import numpy as np
import torch.nn as nn

from model.FastRIFE_Super_Slomo.HSTR_FSS import HSTR_FSS
from model.FastRIFE_Super_Slomo.HSTR_LKSS import HSTR_LKSS



def getting_input(video):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    tot_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # Getting timestamp of each frames
    timestamps = [cap.get(cv2.CAP_PROP_POS_MSEC)]
    proc_timestamps = [0.0]
    frames = []
    while(cap.isOpened()):
        frame_exists, curr_frame = cap.read()
        if(frame_exists):
            timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
            proc_timestamps.append(timestamps[-1] + 1000/fps)
            frames.append(curr_frame)
        else:
            break

    cap.release()

    return frames, proc_timestamps, tot_frame, fps


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='main')
    parser.add_argument('--videos', nargs=2, required=True, type=str, help="Input videos, first HR, then LR video")
    args = parser.parse_args()    

    device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hr_video = args.videos[0]            # Low frame rate video
    lr_video = args.videos[1]            # High frame rate video

    hstr_model = HSTR_FSS()                      # Initializing the model
    # HSTR_FSS.to(device)

    # Processing input videos
    hr_frames, hr_proc_timestamps, hr_tot_frame, hr_fps = getting_input(
        hr_video)
    lr_frames, lr_proc_timestamps, lr_tot_frame, lr_fps = getting_input(
        lr_video)
    
    if(hr_fps > lr_fps):
        print("------------Wrong input order. First HR, then LR video--------------")
        exit(-1)

    # Concatenating input images into one to feed to the network
    #imgs = np.concatenate((lfr_frames[0], lfr_frames[1], hfr_frames[0], hfr_frames[1], hfr_frames[2]), 2)

    padding1_mult = math.floor(hr_frames[0].shape[0] / 32) + 1
    padding2_mult = math.floor(hr_frames[0].shape[1] / 32) + 1
    pad1 = (32 * padding1_mult) - hr_frames[0].shape[0]
    pad2 = (32 * padding2_mult) - hr_frames[0].shape[1]

    # Padding to meet dimension requirements of the network
    # Done before network call, otherwise slows down the network
    padding1 = nn.ReplicationPad2d((0, pad2, pad1, 0))

    # Feeding images to the model
    output = []
    for i in range(len(hr_frames) - 1):
        
        # Moving images to torch tensors
        hr_img0 = torch.from_numpy(np.transpose(hr_frames[i], (2, 0, 1))).to(
                device, non_blocking=True).unsqueeze(0).float() / 255.
        hr_img1 = torch.from_numpy(np.transpose(hr_frames[i + 1], (2, 0, 1))).to(
                device, non_blocking=True).unsqueeze(0).float() / 255.
        lr_img0 = torch.from_numpy(np.transpose(lr_frames[2*i], (2, 0, 1))).to(
                device, non_blocking=True).unsqueeze(0).float() / 255.
        lr_img1 = torch.from_numpy(np.transpose(lr_frames[2*i + 1], (2, 0, 1))).to(
                device, non_blocking=True).unsqueeze(0).float() / 255.
        lr_img2 = torch.from_numpy(np.transpose(lr_frames[2*i + 2], (2, 0, 1))).to(
                device, non_blocking=True).unsqueeze(0).float() / 255.
        
        imgs = torch.cat(
            (hr_img0, hr_img1, lr_img0, lr_img1, lr_img2), 1)
        
        input_imgs = padding1(imgs)
        
        output.append(hr_frames[i])
        
        output_image = hstr_model.inference(input_imgs, lr_proc_timestamps[:3])
        
        padding2 = nn.ReplicationPad2d((0, -pad2, -pad1, 0))
        output_image = padding2(output_image)
        
        result = output_image.detach().numpy()
        result = result[0, :]
        result = np.transpose(result, (1, 2, 0))
        cv2.imshow("win", result)
        cv2.waitKey(100)
        output.append(result)

    height, width, layers = output[0].shape
    size = (width, height)
    four_cc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter('project.mp4', four_cc, 12, size)

    # print(len(output))
    for i in range(len(output)):
        cv2.waitKey(100)
        if str(output[i].dtype) != "uint8":
            temp = output[i]
            output[i] = cv2.normalize(src=temp, dst=None, alpha=0, beta=255,
                                      norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            # cv2.imshow("win", output[i])
            # cv2.waitKey(100)
        out.write(output[i])
    out.release()

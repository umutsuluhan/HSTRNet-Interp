import cv2
import argparse
import numpy as np
import torch
import sys
import os
sys.path.append('.')

from utils.utils import convert_to_torch, padding_vis, convert_to_numpy, crop_inference
from model.HSTRNet import HSTRNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument("--HR_video", type=str, help="Location of high-resolution video")
    parser.add_argument("--LR_video", type=str, help="Location of low-resolution video")
    parser.add_argument("--model_path", default="./pretrained", type=str, help = "Location of pretrained model")
    parser.add_argument("--warping", default="0", type=int)
    args = parser.parse_args()

    model = HSTRNet(device, args.warping)

    model.load_pretrained(args.model_path)
    model.eval()

    HR_video = cv2.VideoCapture(args.HR_video)
    LR_video = cv2.VideoCapture(args.LR_video)

    HR_width = int(HR_video.get(3))
    HR_height = int(HR_video.get(4))
    LR_width = int(LR_video.get(3))
    LR_height = int(LR_video.get(4))
    HR_size = (HR_width, HR_height)
    LR_size = (LR_width, LR_height)
    assert HR_size == LR_size, "You should provide videos with same height and width attributes."
    
    fps_HR = HR_video.get(cv2.CAP_PROP_FPS)
    fps_LR = LR_video.get(cv2.CAP_PROP_FPS)
    assert int(fps_HR) * 2 == int(fps_LR), "Please provide 2x FPS LR video and x FPS HR video."

    HR_frames = []
    LR_frames = []

    while(HR_video.isOpened()):
        ret, frame = HR_video.read()
        if ret == True:
            HR_frames.append(frame)
        else:
            break
    
    while(LR_video.isOpened()):
        ret, frame = LR_video.read()
        if ret == True:
            LR_frames.append(frame)
        else:
            break
    
    high_res_frames = []
    for idx in range(len(HR_frames) - 1):
        with torch.no_grad():
            img0_HR = padding_vis(convert_to_torch(HR_frames[idx]))
            img1_HR = padding_vis(convert_to_torch(HR_frames[idx + 1]))
            img0_LR = padding_vis(convert_to_torch(LR_frames[idx * 2]))
            img1_LR = padding_vis(convert_to_torch(LR_frames[idx * 2 + 1]))
            img2_LR = padding_vis(convert_to_torch(LR_frames[idx * 2 + 2]))

            hr_images = torch.cat((img0_HR, img1_HR), 1)
            lr_imgs = torch.cat((img0_LR, img1_LR, img2_LR), 1)
            pred = model(lr_imgs, hr_images)
            
            pred = crop_inference(pred, HR_height, HR_width)
            img0_HR = crop_inference(img0_HR, HR_height, HR_width)

            high_res_frames.append(convert_to_numpy(img0_HR))
            high_res_frames.append(convert_to_numpy(pred))

    out_frame_h, out_frame_w, _ = high_res_frames[0].shape

    if not os.path.exists("./output/video/"):
        os.makedirs("./output/video/")

    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
    out = cv2.VideoWriter("./output/video/out.mp4", fourcc, fps_LR, (out_frame_w, out_frame_h))
    for frame in(high_res_frames):
        frame = np.uint8(frame)
        out.write(frame)
    
    HR_video.release()
    LR_video.release()
    out.release()
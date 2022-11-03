import sys
sys.path.append('.')
import argparse
import torch
import cv2
import os
import torch.nn.functional as F

from model.HSTRNet import HSTRNet
from utils.utils import convert_to_numpy, convert_to_torch, padding_vis, crop_inference

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument("--image1_LR", type=str, help="Location of first LR image")
    parser.add_argument("--image2_LR", type=str, help="Location of second LR image")
    parser.add_argument("--image3_LR", type=str, help="Location of third LR image")
    parser.add_argument("--image1_HR", required=True, type=str, help="Location of first HR image")
    parser.add_argument("--image2_HR", required=True, type=str, help="Location of third HR image")
    parser.add_argument("--model_path", default="./pretrained", type=str, help = "Location of pretrained model")
    parser.add_argument("--warping", default="0", type=int)
    args = parser.parse_args()

    model = HSTRNet(device, args.warping)

    model.load_pretrained(args.model_path)
    model.eval()
    
    _, _, h, w = convert_to_torch(cv2.imread(args.image1_LR)).shape

    img0_LR = padding_vis(convert_to_torch(cv2.imread(args.image1_LR)))
    img1_LR = padding_vis(convert_to_torch(cv2.imread(args.image2_LR)))
    img2_LR = padding_vis(convert_to_torch(cv2.imread(args.image3_LR)))
    img0_HR = padding_vis(convert_to_torch(cv2.imread(args.image1_HR)))
    img1_HR = padding_vis(convert_to_torch(cv2.imread(args.image2_HR)))
    
    hr_imgs = torch.cat((img0_HR, img1_HR), 1)
    lr_imgs = torch.cat((img0_LR, img1_LR, img2_LR), 1)

    out = crop_inference(model(lr_imgs, hr_imgs), h, w)
    out = convert_to_numpy(out)
    if not os.path.exists("./output/frame/"):
        os.makedirs("./output/frame/")
    cv2.imwrite("./output/frame/pred.png", out)
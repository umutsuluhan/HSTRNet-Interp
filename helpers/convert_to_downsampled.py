import os
import cv2
import torch

import torch.nn.functional as F
import numpy as np
from PIL import Image



device = "cuda"

def convert_to_torch(img):
    result = torch.from_numpy(np.transpose(img, (2, 0, 1))).to(
        device, non_blocking=True).unsqueeze(0).float() / 255.
    
    return result

def interpolate(img):
    result = F.interpolate(img, scale_factor=0.25, mode="bicubic",                # DONT FORGET TO DELETE THIS
                      align_corners=False)
    result = F.interpolate(result, scale_factor=4, mode="bicubic",
                      align_corners=False)
    
    return result
    
def convert_to_np(img):
    result = img.cpu().detach().numpy()
    result = result[0, :]
    result = np.transpose(result, (1, 2, 0))
    
    return result

def normalization(img):
    frame_normed = 255 * (img - img.min()) / (img.max() - img.min())
    frame_normed = np.array(frame_normed, np.int)
    
    return frame_normed

def convert(path):
    img = cv2.imread(path)
    img = convert_to_torch(img)
    img = interpolate(img)
    img = convert_to_np(img)
    img = normalization(img)
    
    return img

def convertion(data_list):

    convertion_list = open(data_list)    

    for entry in convertion_list.readlines():
        entry = entry.replace("\n", "")
        
        img_s = save_path + str(entry)
        
        if not os.path.exists(img_s):
            os.makedirs(img_s)
        
        img0_p = data_path + str(entry) + "/im1.png"
        img0 = convert(img0_p)
        img0_s = img_s + "/im1.png"
        cv2.imwrite(img0_s, img0)

        
        
        img1_p = data_path + str(entry) + "/im2.png"
        img1 = convert(img1_p)
        img1_s = img_s + "/im2.png"
        cv2.imwrite(img1_s, img1)
        
        img2_p = data_path + str(entry) + "/im3.png"
        img2 = convert(img2_p)
        img2_s = img_s + "/im3.png"
        cv2.imwrite(img2_s, img2)

data_path = "/home/hus/Desktop/data/vimeo_triplet/sequences/"                    # SET THIS TO 4X DOWNSAMPLED
save_path = "/home/hus/Desktop/data/vimeo_triplet_lr/sequences/"

test_list = "/home/hus/Desktop/data/vimeo_triplet/tri_testlist.txt"
train_list = "/home/hus/Desktop/data/vimeo_triplet/tri_trainlist.txt"

convertion(test_list)
convertion(train_list)
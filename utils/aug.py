import torchvision.transforms.functional as FF
import random
import numpy as np
from PIL import Image

def horizontal_flip(img0, img1, img2,img3, img4, img5):
    img0 = FF.hflip(img0)
    img1 = FF.hflip(img1)
    img2 = FF.hflip(img2)
    img3 = FF.hflip(img3)
    img4 = FF.hflip(img4)
    img5 = FF.hflip(img5)
    return img0, img1, img2, img3, img4, img5

def rotate(img0, img1, img2, img3, img4, img5):
    degree = random.uniform(-10.0, 10.0)
    rotated_img0 = img0.rotate(degree)
    rotated_img1 = img1.rotate(degree)
    rotated_img2 = img2.rotate(degree)
    rotated_img3 = img3.rotate(degree)
    rotated_img4 = img4.rotate(degree)
    rotated_img5 = img5.rotate(degree)
    return rotated_img0, rotated_img1, rotated_img2, rotated_img3, rotated_img4, rotated_img5

def aug(img0, gt, img1,img0_LR, img1_LR, img2_LR, h, w, mode):
    ih, iw, _ = img0.shape
    x = np.random.randint(0, ih - h + 1)
    y = np.random.randint(0, iw - w + 1)
    img0 = img0[x:x+h, y:y+w, :]
    img1 = img1[x:x+h, y:y+w, :]
    gt = gt[x:x+h, y:y+w, :]
    img0_LR = img0_LR[x:x+h, y:y+w, :]
    img1_LR = img1_LR[x:x+h, y:y+w, :]
    img2_LR = img2_LR[x:x+h, y:y+w, :]
    

    if mode == "train":
        img0_LR = Image.fromarray(img0_LR.astype(np.uint8))
        img1_LR = Image.fromarray(img1_LR.astype(np.uint8))
        img2_LR = Image.fromarray(img2_LR.astype(np.uint8))
        img0 = Image.fromarray(img0.astype(np.uint8))
        gt = Image.fromarray(gt.astype(np.uint8))
        img1 = Image.fromarray(img1.astype(np.uint8))
        

        #Applying horizontal flip with %20 probability 
        p = random.uniform(0.0, 1.0)
        if(p < 0.2):
            img0, gt, img1, img0_LR, img1_LR, img2_LR = horizontal_flip(img0, gt, img1, img0_LR, img1_LR, img2_LR)
        
        #Applying rotation
        img0, gt, img1, img0_LR, img1_LR, img2_LR = rotate(img0, gt, img1, img0_LR, img1_LR, img2_LR)
        
        img0 = np.array(img0)
        gt = np.array(gt)
        img1 = np.array(img1)
        img0_LR = np.array(img0_LR)
        img1_LR = np.array(img1_LR)
        img2_LR = np.array(img2_LR)

    return img0, gt, img1, img0_LR, img1_LR, img2_LR
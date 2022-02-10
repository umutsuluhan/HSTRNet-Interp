import cv2
import torch
import numpy as np
import torch.nn.functional as F
import random
from torch.utils.data import DataLoader, Dataset
import os
import torchvision.transforms.functional as FF
from PIL import Image
import time
import math
from model.pytorch_msssim import ssim

device = "cuda:2"

def image_show(result):
    result = result.cpu().detach().numpy()
    result = result[0, :]
    result = np.transpose(result, (1, 2, 0))
    cv2.imshow("win", result)
    cv2.waitKey(2000)

#-------------------------------------------------------------------
def homography(img, p):        
    homography = np.zeros((3,3))
    homography[0][0] = 1
    homography[1][1] = 1
    homography[2][2] = 1

    #homography = homography + (0.00000000001 ** 0.5) * np.random.randn(homography.shape[0], homography.shape[1])
    homography = homography + (p ** 0.5) * np.random.randn(homography.shape[0], homography.shape[1])
    homography[2][2] = 1

    homography_img = cv2.warpPerspective(img, homography, (img.shape[1], img.shape[0]))
    
    """test_homography_img = torch.from_numpy(np.transpose(homography_img, (2, 0, 1))).to(
        device, non_blocking=True).unsqueeze(0).float() / 255.
    test_img = torch.from_numpy(np.transpose(img, (2, 0, 1))).to(
        device, non_blocking=True).unsqueeze(0).float() / 255.
    psnr = -10 * math.log10(((test_homography_img - test_img) * (test_homography_img - test_img)).mean())
    ssim_ = float(ssim(test_homography_img, test_img))
    
    print("Homography psnr:" + str(psnr))
    print("Homography ssim:" + str(ssim_))"""
    
    return homography_img
#-------------------------------------------------------------------

#-------------------------------------------------------------------
def gaussian_noise(img0, img1, img2):  
    """test_img = torch.from_numpy(np.transpose(img0, (2, 0, 1))).to(
        device, non_blocking=True).unsqueeze(0).float() / 255."""
    
    factor = random.uniform(0.0, 2.5)
    img0 = img0 + np.random.randn(*img0.shape) * factor + 0
    img1 = img1 + np.random.randn(*img1.shape) * factor + 0
    img2 = img2 + np.random.randn(*img2.shape) * factor + 0
    
    """test_gaussian_img = torch.from_numpy(np.transpose(img0, (2, 0, 1))).to(
        device, non_blocking=True).unsqueeze(0).float() / 255.
    
    psnr = -10 * math.log10(((test_gaussian_img - test_img) * (test_gaussian_img - test_img)).mean())
    ssim_ = float(ssim(test_gaussian_img, test_img))
    
    print("Gaussian psnr:" + str(psnr))
    print("Gaussian ssim:" + str(ssim_))"""
    
    return img0, img1, img2
#-------------------------------------------------------------------

#-------------------------------------------------------------------
def contrast(img0, img1, img2):  
    """test_img = torch.from_numpy(np.transpose(img0, (2, 0, 1))).to(
        device, non_blocking=True).unsqueeze(0).float() / 255."""
    
    factor = random.uniform(0.9, 1.1)
    img0 = FF.adjust_contrast(img0, factor)
    img1 = FF.adjust_contrast(img1, factor)
    img2 = FF.adjust_contrast(img2, factor)
    
    """img0 = np.array(img0)
    test_contrast_img = torch.from_numpy(np.transpose(img0, (2, 0, 1))).to(
        device, non_blocking=True).unsqueeze(0).float() / 255.
    
    psnr = -10 * math.log10(((test_contrast_img - test_img) * (test_contrast_img - test_img)).mean())
    ssim_ = float(ssim(test_contrast_img, test_img))
    
    print("Contrast psnr:" + str(psnr))
    print("Contrast ssim:" + str(ssim_))"""
    
    return img0, img1, img2
#-------------------------------------------------------------------

#-------------------------------------------------------------------
def horizontal_flip(img0, img1, img2,img3, img4, img5):
    img0 = FF.hflip(img0)
    img1 = FF.hflip(img1)
    img2 = FF.hflip(img2)
    img3 = FF.hflip(img3)
    img4 = FF.hflip(img4)
    img5 = FF.hflip(img5)
    return img0, img1, img2, img3, img4, img5
#-------------------------------------------------------------------

#-------------------------------------------------------------------
def rotate(img0, img1, img2, img3, img4, img5):
    degree = random.uniform(-10.0, 10.0)
    rotated_img0 = img0.rotate(degree)
    rotated_img1 = img1.rotate(degree)
    rotated_img2 = img2.rotate(degree)
    rotated_img3 = img3.rotate(degree)
    rotated_img4 = img4.rotate(degree)
    rotated_img5 = img5.rotate(degree)
    return rotated_img0, rotated_img1, rotated_img2, rotated_img3, rotated_img4, rotated_img5
#-------------------------------------------------------------------


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
        """# Applying homography to LR frames
        #-------------------------------------------------------------------
        img0_LR = homography(img0_LR, 0.00000000001)
        img1_LR = homography(img1_LR, 0.0000000005)
        img2_LR = homography(img2_LR, 0.00000000001)
        #-------------------------------------------------------------------

        #Applying gaussian noise to HR and LR frames
        #-------------------------------------------------------------------
        img0_LR, img1_LR, img2_LR = gaussian_noise(img0_LR, img1_LR, img2_LR)
        #-------------------------------------------------------------------

        img0_LR = Image.fromarray(img0_LR.astype(np.uint8))
        img1_LR = Image.fromarray(img1_LR.astype(np.uint8))
        img2_LR = Image.fromarray(img2_LR.astype(np.uint8))
        img0 = Image.fromarray(img0.astype(np.uint8))
        gt = Image.fromarray(gt.astype(np.uint8))
        img1 = Image.fromarray(img1.astype(np.uint8))
        

        #Applying contrast to HR and LR frames
        #-------------------------------------------------------------------
        img0_LR, img1_LR, img2_LR = contrast(img0_LR, img1_LR, img2_LR)
        #-------------------------------------------------------------------"""
        
        img0_LR = Image.fromarray(img0_LR.astype(np.uint8))
        img1_LR = Image.fromarray(img1_LR.astype(np.uint8))
        img2_LR = Image.fromarray(img2_LR.astype(np.uint8))
        img0 = Image.fromarray(img0.astype(np.uint8))
        gt = Image.fromarray(gt.astype(np.uint8))
        img1 = Image.fromarray(img1.astype(np.uint8))
        

        #Applying horizontal flip with %20 probability 
        #-------------------------------------------------------------------
        p = random.uniform(0.0, 1.0)
        if(p < 0.2):
            img0, gt, img1, img0_LR, img1_LR, img2_LR = horizontal_flip(img0, gt, img1, img0_LR, img1_LR, img2_LR)
        #-------------------------------------------------------------------

        #Applying rotation
        #-------------------------------------------------------------------
        img0, gt, img1, img0_LR, img1_LR, img2_LR = rotate(img0, gt, img1, img0_LR, img1_LR, img2_LR)
        #-------------------------------------------------------------------

        img0 = np.array(img0)
        gt = np.array(gt)
        img1 = np.array(img1)
        img0_LR = np.array(img0_LR)
        img1_LR = np.array(img1_LR)
        img2_LR = np.array(img2_LR)

    return img0, gt, img1, img0_LR, img1_LR, img2_LR

class VimeoDataset(Dataset):
    def __init__(self, dataset_name, data_root, device, batch_size=32):
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.data_root = data_root
        self.device = device
        self.load_data()
        self.h = 256
        self.w = 448
        xx = np.arange(0, self.w).reshape(1,-1).repeat(self.h,0)
        yy = np.arange(0, self.h).reshape(-1,1).repeat(self.w,1)
        self.grid = np.stack((xx,yy),2).copy()

    def __len__(self):
        return len(self.meta_data_HR)

    def load_data(self):
        self.trainlist_HR = []
        self.trainlist_LR = []
        self.testlist_HR = []
        self.testlist_LR = []
        data_path_HR = os.path.join(self.data_root, "sequences/")
        data_path_LR = os.path.join(self.data_root, "vimeo_triplet_lr/sequences/")
        train_path = os.path.join(self.data_root, 'tri_trainlist.txt')
        test_path = os.path.join(self.data_root, 'tri_testlist.txt')
        with open(train_path, 'r') as f:
            self.trainlist_HR = f.read().splitlines()
        with open(train_path, 'r') as f:
            self.trainlist_LR = f.read().splitlines()
        with open(test_path, 'r') as f:
            self.testlist_HR = f.read().splitlines()
        with open(test_path, 'r') as f:
            self.testlist_LR = f.read().splitlines()

        for i, entry in enumerate(self.trainlist_HR):
            new_entry = data_path_HR  + entry
            self.trainlist_HR[i] = new_entry
        for i, entry in enumerate(self.trainlist_LR):
            new_entry = data_path_LR  + entry
            self.trainlist_LR[i] = new_entry
        for i, entry in enumerate(self.testlist_HR):
            new_entry = data_path_HR  + entry
            self.testlist_HR[i] = new_entry
        for i, entry in enumerate(self.testlist_LR):
            new_entry = data_path_LR  + entry
            self.testlist_LR[i] = new_entry
        # SHOULD REMOVE THIS FOR BENCHMARK
        # -------------------------------------------------------------------
        self.testlist_HR_train = []
        self.testlist_LR_train = []
        for i, entry in enumerate(self.testlist_HR):
            if(i % 5 == 0):
                new_entry = data_path_HR + entry
                self.testlist_HR_train.append(new_entry)
        for i, entry in enumerate(self.testlist_LR):
            if(i % 5 == 0):
                new_entry = data_path_LR + entry
                self.testlist_LR_train.append(new_entry)
        # -------------------------------------------------------------------
        if self.dataset_name == 'train':
            self.meta_data_HR = self.trainlist_HR
            self.meta_data_LR = self.trainlist_LR
            print('Number of training samples in: ' + str(self.data_root.split("/")[-1]), len(self.meta_data_HR))
        else:
            self.meta_data_HR = self.testlist_HR
            self.meta_data_LR = self.testlist_LR
            print('Number of validation samples in: ' + str(self.data_root.split("/")[-1]), len(self.meta_data_HR))
        self.nr_sample = len(self.meta_data_HR)

    
    def getimg(self, index):
        img_path_HR = self.meta_data_HR[index]
        imgpaths_HR = [img_path_HR + '/im1.png', img_path_HR + '/im2.png', img_path_HR + '/im3.png']
        
        img_path_LR = self.meta_data_LR[index]
        imgpaths_LR = [img_path_LR + '/im1.png', img_path_LR + '/im2.png', img_path_LR + '/im3.png']

        img0_HR = cv2.imread(imgpaths_HR[0])
        gt = cv2.imread(imgpaths_HR[1])
        img1_HR = cv2.imread(imgpaths_HR[2])
        
        img0_LR = cv2.imread(imgpaths_LR[0])
        img1_LR = cv2.imread(imgpaths_LR[1])
        img2_LR = cv2.imread(imgpaths_LR[2])


        return img0_HR, gt, img1_HR, img0_LR, img1_LR, img2_LR


    def __getitem__(self, index):
        img0_HR, gt, img1_HR, img0_LR, img1_LR, img2_LR = self.getimg(index)

        if self.dataset_name == 'train':

            img0_HR, gt, img1_HR, img0_LR, img1_LR, img2_LR = aug(img0_HR, gt, img1_HR,img0_LR, img1_LR, img2_LR, 128, 128, "train")
            img0_HR = torch.from_numpy(img0_HR.copy()).permute(2, 0, 1)
            img1_HR = torch.from_numpy(img1_HR.copy()).permute(2, 0, 1)
            gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
            img0_LR = torch.from_numpy(img0_LR.copy()).permute(2, 0, 1)
            img1_LR = torch.from_numpy(img1_LR.copy()).permute(2, 0, 1)
            img2_LR = torch.from_numpy(img2_LR.copy()).permute(2, 0, 1)
            return torch.cat((img0_HR, img1_HR, gt, img0_LR, img1_LR, img2_LR), 0)
        elif self.dataset_name == "validation":
            img0_HR, gt, img1_HR, img0_LR, img1_LR, img2_LR = aug(img0_HR, gt, img1_HR,img0_LR, img1_LR, img2_LR, 256, 448, "test")
            img0_HR = torch.from_numpy(img0_HR.copy()).permute(2, 0, 1).to(self.device)
            img1_HR = torch.from_numpy(img1_HR.copy()).permute(2, 0, 1).to(self.device)
            gt = torch.from_numpy(gt.copy()).permute(2, 0, 1).to(self.device)
            img0_LR = torch.from_numpy(img0_LR.copy()).permute(2, 0, 1).to(self.device)
            img1_LR = torch.from_numpy(img1_LR.copy()).permute(2, 0, 1).to(self.device)
            img2_LR = torch.from_numpy(img2_LR.copy()).permute(2, 0, 1).to(self.device)
            return torch.cat((img0_HR, img1_HR, gt, img0_LR, img1_LR, img2_LR), 0)

       

class VizdroneDataset(Dataset):
    def __init__(self, dataset_name, data_root, device, batch_size=32):
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.data_root = data_root
        self.device = device
        self.load_data()
        self.h = 256
        self.w = 448


    def __len__(self):
        return len(self.meta_data_HR) - 3
    
    def load_data(self):
        self.trainlist_HR = []
        self.trainlist_LR = []
        self.testlist_HR = []
        self.testlist_LR = []
        data_path_HR_train = os.path.join(self.data_root, "train/HR/")
        data_path_LR_train = os.path.join(self.data_root, "vizdrone_lr/train/")
        data_path_HR_val = os.path.join(self.data_root, "val/HR/")
        data_path_LR_val = os.path.join(self.data_root, "vizdrone_lr/val/")
        train_path = os.path.join(self.data_root, 'original_train_list.csv')
        test_path = os.path.join(self.data_root, 'original_val_list.csv')
        with open(train_path, 'r') as f:
            self.trainlist_HR = f.read().splitlines()
        with open(train_path, 'r') as f:
            self.trainlist_LR = f.read().splitlines()
        with open(test_path, 'r') as f:
            self.testlist_HR = f.read().splitlines()
        with open(test_path, 'r') as f:
            self.testlist_LR = f.read().splitlines()
        
        for i, entry in enumerate(self.trainlist_HR):
            new_entry = data_path_HR_train  + entry + ".png"
            self.trainlist_HR[i] = new_entry
        for i, entry in enumerate(self.trainlist_LR):
            new_entry = data_path_LR_train  + entry + ".png"
            self.trainlist_LR[i] = new_entry
        for i, entry in enumerate(self.testlist_HR):
            new_entry = data_path_HR_val + entry + ".png"
            self.testlist_HR[i] = new_entry
        for i, entry in enumerate(self.testlist_LR):
            new_entry = data_path_LR_val + entry + ".png"
            self.testlist_LR[i] = new_entry

        if self.dataset_name == 'train':
            self.meta_data_HR = self.trainlist_HR
            self.meta_data_LR = self.trainlist_LR
            print('Number of training samples in: ' + str(self.data_root.split("/")[-1]), len(self.meta_data_HR))
        else:
            self.meta_data_HR = self.testlist_HR
            self.meta_data_LR = self.testlist_LR
            print('Number of validation samples in: ' + str(self.data_root.split("/")[-1]), len(self.meta_data_HR))
        self.nr_sample = len(self.meta_data_HR)
        
    def getimg(self, index):
        img0_HR = cv2.imread(self.meta_data_HR[index])
        gt = cv2.imread(self.meta_data_HR[index + 1])
        img1_HR = cv2.imread(self.meta_data_HR[index + 2])
        
        img0_LR = cv2.imread(self.meta_data_LR[index])
        img1_LR = cv2.imread(self.meta_data_LR[index + 1])
        img2_LR = cv2.imread(self.meta_data_LR[index + 2])

        return img0_HR, gt, img1_HR, img0_LR, img1_LR, img2_LR

    def __getitem__(self, index):
        img0_HR, gt, img1_HR, img0_LR, img1_LR, img2_LR = self.getimg(index)
    
        if self.dataset_name == 'train':
            img0_HR, gt, img1_HR, img0_LR, img1_LR, img2_LR = aug(img0_HR, gt, img1_HR,img0_LR, img1_LR, img2_LR, 128, 128, "train")
            img0_HR = torch.from_numpy(img0_HR.copy()).permute(2, 0, 1)
            img1_HR = torch.from_numpy(img1_HR.copy()).permute(2, 0, 1)
            gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
            img0_LR = torch.from_numpy(img0_LR.copy()).permute(2, 0, 1)
            img1_LR = torch.from_numpy(img1_LR.copy()).permute(2, 0, 1)
            img2_LR = torch.from_numpy(img2_LR.copy()).permute(2, 0, 1)
            return torch.cat((img0_HR, img1_HR, gt, img0_LR, img1_LR, img2_LR), 0)
        elif self.dataset_name == "validation":
            img0_HR, gt, img1_HR, img0_LR, img1_LR, img2_LR = aug(img0_HR, gt, img1_HR,img0_LR, img1_LR, img2_LR, 256, 448, "test")
            img0_HR = torch.from_numpy(img0_HR.copy()).permute(2, 0, 1).to(self.device)
            img1_HR = torch.from_numpy(img1_HR.copy()).permute(2, 0, 1).to(self.device)
            gt = torch.from_numpy(gt.copy()).permute(2, 0, 1).to(self.device)
            img0_LR = torch.from_numpy(img0_LR.copy()).permute(2, 0, 1).to(self.device)
            img1_LR = torch.from_numpy(img1_LR.copy()).permute(2, 0, 1).to(self.device)
            img2_LR = torch.from_numpy(img2_LR.copy()).permute(2, 0, 1).to(self.device)
            return torch.cat((img0_HR, img1_HR, gt, img0_LR, img1_LR, img2_LR), 0)

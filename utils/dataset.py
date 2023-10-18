import cv2
import torch
import numpy as np
import torch.nn.functional as F
import random
from torch.utils.data import DataLoader, Dataset
import os
import math
import torch.nn as nn

from utils.utils import image_show
from utils.aug import aug

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VimeoDataset(Dataset):
    def __init__(self, dataset_name, data_root, device, batch_size=32):
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.data_root = data_root
        self.device = device
        self.load_data()
        self.h = 256
        self.w = 448

    def __len__(self):
        return len(self.meta_data_HR)

    def load_data(self):
        # Reading and storing image paths.
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
        # Storing image into a numpy array
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

        # Training partition process
        if self.dataset_name == 'train':
            # Augmenting size to 128x128 and moving images to torch.tensor
            img0_HR, gt, img1_HR, img0_LR, img1_LR, img2_LR = aug(img0_HR, gt, img1_HR,img0_LR, img1_LR, img2_LR, 128, 128, "train")
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
        data_path_LR_val = os.path.join(self.data_root,"val/LR/")
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
            new_entry = data_path_HR_train  + entry + ".jpg"
            self.trainlist_HR[i] = new_entry
        for i, entry in enumerate(self.trainlist_LR):
            new_entry = data_path_LR_train  + entry + ".jpg"
            self.trainlist_LR[i] = new_entry
        for i, entry in enumerate(self.testlist_HR):
            new_entry = data_path_HR_val + entry + ".jpg"
            self.testlist_HR[i] = new_entry
        for i, entry in enumerate(self.testlist_LR):
            new_entry = data_path_LR_val + entry + ".jpg"
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
        
    def padding(self, img):
        padding1_mult = math.floor(img.shape[1] / 32) + 1
        padding2_mult = math.floor(img.shape[2] / 32) + 1
        pad1 = (32 * padding1_mult) - img.shape[1]
        pad2 = (32 * padding2_mult) - img.shape[2]

        img = torch.unsqueeze(img, 0)
        padding = nn.ZeroPad2d((int(pad2/2), int(pad2/2), int(pad1/2), int(pad1/2)))
        img = img.float()
        img = padding(img)
        img = torch.squeeze(img, 0)
        
        return img

    def getimg(self, index):
        img0_HR = cv2.imread(self.meta_data_HR[index])
        gt = cv2.imread(self.meta_data_HR[index + 1])
        img1_HR = cv2.imread(self.meta_data_HR[index + 2])
        
        img0_LR = cv2.imread(self.meta_data_LR[index])
        img1_LR = cv2.imread(self.meta_data_LR[index + 1])
        img2_LR = cv2.imread(self.meta_data_LR[index + 2])
        return img0_HR, gt, img1_HR, img0_LR, img1_LR, img2_LR

    def __getitem__(self, index):
        # Images are padded to meet FusionNet size requirements
        img0_HR, gt, img1_HR, img0_LR, img1_LR, img2_LR = self.getimg(index)
        img0_HR = self.padding(torch.from_numpy(img0_HR.copy()).permute(2, 0, 1).to(self.device))
        img1_HR = self.padding(torch.from_numpy(img1_HR.copy()).permute(2, 0, 1).to(self.device))
        gt = self.padding(torch.from_numpy(gt.copy()).permute(2, 0, 1).to(self.device))
        img0_LR = self.padding(torch.from_numpy(img0_LR.copy()).permute(2, 0, 1).to(self.device))
        img1_LR = self.padding(torch.from_numpy(img1_LR.copy()).permute(2, 0, 1).to(self.device))
        img2_LR = self.padding(torch.from_numpy(img2_LR.copy()).permute(2, 0, 1).to(self.device))
        return torch.cat((img0_HR, img1_HR, gt, img0_LR, img1_LR, img2_LR), 0)
import cv2
import torch
import numpy as np
import torch.nn.functional as F
import random
from torch.utils.data import DataLoader, Dataset
import os

cv2.setNumThreads(1)
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
            new_entry = data_path_HR + entry
            self.testlist_HR[i] = new_entry
        for i, entry in enumerate(self.testlist_LR):
            new_entry = data_path_LR + entry
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

    def aug(self, img0, gt, img1,img0_LR, img1_LR, img2_LR, h, w):
        ih, iw, _ = img0.shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        img0 = img0[x:x+h, y:y+w, :]
        img1 = img1[x:x+h, y:y+w, :]
        gt = gt[x:x+h, y:y+w, :]
        img0_LR = img0_LR[x:x+h, y:y+w, :]
        img1_LR = img1_LR[x:x+h, y:y+w, :]
        img2_LR = img2_LR[x:x+h, y:y+w, :]
        return img0, gt, img1, img0_LR, img1_LR, img2_LR

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

            img0_HR, gt, img1_HR, img0_LR, img1_LR, img2_LR = self.aug(img0_HR, gt, img1_HR,img0_LR, img1_LR, img2_LR, 128, 128)
            
            img0_HR = torch.from_numpy(img0_HR.copy()).permute(2, 0, 1)
            img1_HR = torch.from_numpy(img1_HR.copy()).permute(2, 0, 1)
            gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)

            img0_LR = torch.from_numpy(img0_LR.copy()).permute(2, 0, 1)
            img1_LR = torch.from_numpy(img1_LR.copy()).permute(2, 0, 1)
            img2_LR = torch.from_numpy(img2_LR.copy()).permute(2, 0, 1)
            return torch.cat((img0_HR, img1_HR, gt, img0_LR, img1_LR, img2_LR), 0)
            if random.uniform(0, 1) < 0.5:
                img0_HR = img0_HR[:, :, ::-1]
                img1_HR = img1_HR[:, :, ::-1]
                gt = gt[:, :, ::-1]
                img0_LR = img0_LR[:, :, ::-1]
                img1_LR = img1_LR[:, :, ::-1]
                img2_LR = img2_LR[:, :, ::-1]
            if random.uniform(0, 1) < 0.5:
                img0_HR = img0_HR[::-1]
                img1_HR = img1_HR[::-1]
                gt = gt[::-1]
                img0_LR = img0_LR[::-1]
                img1_LR = img1_LR[::-1]
                img2_LR = img2_LR[::-1]
            if random.uniform(0, 1) < 0.5:
                img0_HR = img0_HR[:, ::-1]
                img1_HR = img1_HR[:, ::-1]
                gt = gt[:, ::-1]
                img0_LR = img0_LR[:, ::-1]
                img1_LR = img1_LR[:, ::-1]
                img2_LR = img2_LR[:, ::-1]
            if random.uniform(0, 1) < 0.5:
                tmp = img1_HR
                img1_HR = img0_HR
                img0_HR = tmp
                
                tmp = img1_LR
                img1_LR = img0_LR
                img0_LR = tmp
        elif self.dataset_name == "validation":
            img0_HR, gt, img1_HR, img0_LR, img1_LR, img2_LR = self.aug(img0_HR, gt, img1_HR,img0_LR, img1_LR, img2_LR, 256, 448)
        img0_HR = torch.from_numpy(img0_HR.copy()).permute(2, 0, 1).to(self.device)
        img1_HR = torch.from_numpy(img1_HR.copy()).permute(2, 0, 1).to(self.device)
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1).to(self.device)
        img0_LR = torch.from_numpy(img0_LR.copy()).permute(2, 0, 1).to(self.device)
        img1_LR = torch.from_numpy(img1_LR.copy()).permute(2, 0, 1).to(self.device)
        img2_LR = torch.from_numpy(img2_LR.copy()).permute(2, 0, 1).to(self.device)
        return torch.cat((img0_HR, img1_HR, gt, img0_LR, img1_LR, img2_LR), 0)

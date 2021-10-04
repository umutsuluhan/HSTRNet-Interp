import cv2
import torch
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset
import os

cv2.setNumThreads(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class VimeoDataset(Dataset):
    def __init__(self, dataset_name, data_root, batch_size=32):
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.data_root = data_root
        self.load_data()
        self.h = 256
        self.w = 448
        xx = np.arange(0, self.w).reshape(1,-1).repeat(self.h,0)
        yy = np.arange(0, self.h).reshape(-1,1).repeat(self.w,1)
        self.grid = np.stack((xx,yy),2).copy()

    def __len__(self):
        return len(self.meta_data)

    def load_data(self):
        """self.train_paths = []
        self.val_paths = []
        """
        self.trainlist = []
        self.testlist = []
        # data_root = '../../../../ortak/mughees/datasets/vimeo_triplet'
        #data_root = '/home/hus/Desktop/data/vimeo_triplet'
        data_path = os.path.join(self.data_root, "sequences/")
        train_path = os.path.join(self.data_root, 'tri_trainlist.txt')
        test_path = os.path.join(self.data_root, 'tri_testlist.txt')
        with open(train_path, 'r') as f:
            self.trainlist = f.read().splitlines()
        with open(test_path, 'r') as f:
            self.testlist = f.read().splitlines()

        for i, entry in enumerate(self.trainlist):
            new_entry = data_path  + entry
            self.trainlist[i] = new_entry
        for i, entry in enumerate(self.testlist):
            new_entry = data_path + entry
            self.testlist[i] = new_entry

        if self.dataset_name == 'train':
            self.meta_data = self.trainlist
            print('Number of training samples in:' + str(self.data_root.split("/")[-1]), len(self.meta_data))
        else:
            self.meta_data = self.testlist
            print('Number of validation samples in:' + str(self.data_root.split("/")[-1]), len(self.meta_data))
        self.nr_sample = len(self.meta_data)

    def aug(self, img0, gt, img1, h, w):
        ih, iw, _ = img0.shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        img0 = img0[x:x+h, y:y+w, :]
        img1 = img1[x:x+h, y:y+w, :]
        gt = gt[x:x+h, y:y+w, :]
        return img0, gt, img1

    def getimg(self, index):
        img_path = self.meta_data[index]
        imgpaths = [img_path + '/im1.png', img_path + '/im2.png', img_path + '/im3.png']

        img0 = cv2.imread(imgpaths[0])
        gt = cv2.imread(imgpaths[1])
        img1 = cv2.imread(imgpaths[2])
        return img0, gt, img1

    def __getitem__(self, index):
        img0, gt, img1 = self.getimg(index)

        if self.dataset_name == 'train':
            img0, gt, img1 = self.aug(img0, gt, img1, 256, 448)
            img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1)
            img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
            gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
            return torch.cat((img0, img1, gt), 0)
            if random.uniform(0, 1) < 0.5:
                img0 = img0[:, :, ::-1]
                img1 = img1[:, :, ::-1]
                gt = gt[:, :, ::-1]
            if random.uniform(0, 1) < 0.5:
                img0 = img0[::-1]
                img1 = img1[::-1]
                gt = gt[::-1]
            if random.uniform(0, 1) < 0.5:
                img0 = img0[:, ::-1]
                img1 = img1[:, ::-1]
                gt = gt[:, ::-1]
            if random.uniform(0, 1) < 0.5:
                tmp = img1
                img1 = img0
                img0 = tmp
        img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1)
        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
        return torch.cat((img0, img1, gt), 0)

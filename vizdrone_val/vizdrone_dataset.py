import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader 
from PIL import Image
import torchvision.transforms.functional as F
from helper import show_figure

class LRtoHR_loader(torch.utils.data.Dataset):
    def __init__(self, root, train_list, n_f = 3):
        self.base_dir = root + 'LR/'
        self.hr_dir = root + 'HR/'
        self.train_list = train_list

        self.n_f = n_f #neighbouring_frames
        self.pathlist = self.loadpath(self.train_list)

    def __getitem__(self, index):
        frames = []
        frame = self.pathlist[index]
        video_folder,frame_label = frame.split('/')
        range_frame = int(frame_label[2:]) #removing 'im' part from image caption and making it int
        
        for i in range(range_frame-self.n_f,range_frame+self.n_f + 1): #range(-2,2+1): -2,-1,0,1,2
            frames.append(np.array(Image.open(os.path.join(self.base_dir, video_folder, f"{i:07d}"+'.jpg')),dtype = 'float32')) # load images with noise.
        
        frames = np.transpose(frames, (0, 3, 1, 2))/255.0

        frame_gt = np.array(Image.open(os.path.join(self.hr_dir, frame + '.jpg')),dtype = 'float32')  #load ground truth
        frame_gt = np.transpose(frame_gt,(2, 0, 1))/255.0

        return torch.from_numpy(frames) ,torch.from_numpy(frame_gt), frame # , torch.from_numpy(frame_hr),

    def __len__(self):
        return len(self.pathlist)
       
    def loadpath(self, pathlistfile):
        fp = open(pathlistfile)
        pathlist = fp.read().splitlines()
        fp.close()
        return pathlist

class LRtoHR_with_HR_Loader(torch.utils.data.Dataset):
    def __init__(self, root, train_list, n_f = 3):
        self.base_dir = root + 'LR/'
        self.hr_dir = root + 'HR/'

        self.n_f = n_f #neighbouring_frames
        self.pathlist = self.loadpath(train_list)

    def __getitem__(self, index):
        frames = []
        frame = self.pathlist[index]
        video_folder,frame_label = frame.split('/')
        range_frame = int(frame_label[2:]) #removing 'im' part from image caption and making it int
        
        for i in range(range_frame-self.n_f,range_frame+self.n_f + 1): #range(-2,2+1): -2,-1,0,1,2
            frames.append(np.array(Image.open(os.path.join(self.base_dir, video_folder, f"{i:07d}"+'.jpg')),dtype = 'float32')) # load images with noise.
        
        hr_idx =range_frame + 1
        frames.append(np.array(Image.open(os.path.join(self.hr_dir, video_folder, f"{hr_idx:07d}"+'.jpg')),dtype = 'float32'))
        frames = np.transpose(frames, (0, 3, 1, 2))/255.0

        #print(self.hr_dir, video_folder, f"{hr_idx:07d}"+'.jpg')
        frame_gt = np.array(Image.open(os.path.join(self.hr_dir, frame + '.jpg')),dtype = 'float32') #load ground truth
        frame_gt = np.transpose(frame_gt,(2, 0, 1))/255.0


        return torch.from_numpy(frames), torch.from_numpy(frame_gt), frame

    def __len__(self):
        return len(self.pathlist)
    
    def loadpath(self, pathlistfile):
        fp = open(pathlistfile)
        pathlist = fp.read().splitlines()
        fp.close()
        return pathlist

class LRtoHR_with_HR_Loader_new_terminology(torch.utils.data.Dataset):
    def __init__(self, root, train_list, n_f = 1):
        self.base_dir = root + 'LR/'
        self.hr_dir = root + 'HR/'

        self.n_f = n_f #neighbouring_frames
        self.pathlist = self.loadpath(train_list)

    def __getitem__(self, index):
        frames = [] 
        frame = self.pathlist[index]
        video_folder,frame_label = frame.split('/')
        range_frame = int(frame_label[2:]) #removing 'im' part from image caption and making it int
        
        for i in range(range_frame-self.n_f,range_frame+self.n_f + 1): #range(-2,2+1): -2,-1,0,1,2
            frames.append(np.array(Image.open(os.path.join(self.base_dir, video_folder, f"{i:07d}"+'.jpg')),dtype = 'float32')) # load images with noise.
        
        #hr_idx =range_frame + 1
        
        frames.append(np.array(Image.open(os.path.join(self.hr_dir, video_folder, f"{(range_frame - 1):07d}"+'.jpg')),dtype = 'float32'))
        frames.append(np.array(Image.open(os.path.join(self.hr_dir, video_folder, f"{(range_frame + 1):07d}"+'.jpg')),dtype = 'float32'))
        
        frames = np.transpose(frames, (0, 3, 1, 2))/255.0
        #print(frames.shape)
        frames = frames[[0,2,1,3,4]]
        #print(self.hr_dir, video_folder, f"{hr_idx:07d}"+'.jpg')
        frame_gt = np.array(Image.open(os.path.join(self.hr_dir, frame + '.jpg')),dtype = 'float32') #load ground truth
        frame_gt = np.transpose(frame_gt,(2, 0, 1))/255.0


        return torch.from_numpy(frames), torch.from_numpy(frame_gt), frame

    def __len__(self):
        return len(self.pathlist)
    
    def loadpath(self, pathlistfile):
        fp = open(pathlistfile)
        pathlist = fp.read().splitlines()
        fp.close()
        return pathlist

class HR_HR_interpolation(torch.utils.data.Dataset):
    def __init__(self, root, train_list, n_f = 1):
        self.base_dir = root + 'LR/'
        self.hr_dir = root + 'HR/'

        self.n_f = n_f #neighbouring_frames
        self.pathlist = self.loadpath(train_list)

    def __getitem__(self, index):
        frames = [] 
        frame = self.pathlist[index]
        video_folder,frame_label = frame.split('/')
        range_frame = int(frame_label[2:]) #removing 'im' part from image caption and making it int
        
        # for i in range(range_frame-self.n_f,range_frame+self.n_f + 1): #range(-2,2+1): -2,-1,0,1,2
        #     frames.append(np.array(Image.open(os.path.join(self.base_dir, video_folder, f"{i:07d}"+'.jpg')),dtype = 'float32')) # load images with noise.
        
        #hr_idx =range_frame + 1
        
        frames.append(np.array(Image.open(os.path.join(self.hr_dir, video_folder, f"{(range_frame - 1):07d}"+'.jpg')),dtype = 'float32'))
        frames.append(np.array(Image.open(os.path.join(self.hr_dir, video_folder, f"{(range_frame + 1):07d}"+'.jpg')),dtype = 'float32'))
        
        frames = np.transpose(frames, (0, 3, 1, 2))/255.0
        #print(frames.shape)
        # frames = frames[[0,2,1,3,4]]
        #print(self.hr_dir, video_folder, f"{hr_idx:07d}"+'.jpg')
        frame_gt = np.array(Image.open(os.path.join(self.hr_dir, frame + '.jpg')),dtype = 'float32') #load ground truth
        frame_gt = np.transpose(frame_gt,(2, 0, 1))/255.0


        return torch.from_numpy(frames), torch.from_numpy(frame_gt), frame

    def __len__(self):
        return len(self.pathlist)
    
    def loadpath(self, pathlistfile):
        fp = open(pathlistfile)
        pathlist = fp.read().splitlines()
        fp.close()
        return pathlist



#NOT OPTIMIZED

# class LRtoHR_with_HR_Loader(torch.utils.data.Dataset):
#     def __init__(self, root, train_list, n_f = 3):
#         self.base_dir = root + 'LR/'
#         self.hr_dir = root + 'HR/'

#         self.n_f = n_f #neighbouring_frames
#         self.pathlist = self.loadpath(train_list)

#     def __getitem__(self, index):
#         frames = []
#         frame = self.pathlist[index]
#         video_folder,frame_label = frame.split('/')
#         range_frame = int(frame_label[2:]) #removing 'im' part from image caption and making it int
        
#         for i in range(range_frame-self.n_f,range_frame+self.n_f + 1): #range(-2,2+1): -2,-1,0,1,2
#             frames.append(np.array(Image.open(os.path.join(self.base_dir, video_folder, f"{i:07d}"+'.jpg')))) # load images with noise.

#         frames = np.transpose(frames, (0, 3, 1, 2))/255.0

#         #hr_idx = ((range_frame+self.n_f)//7)*7
#         hr_idx =range_frame + 1
#         frame_hr = np.array(Image.open(os.path.join(self.hr_dir, video_folder, f"{hr_idx:07d}"+'.jpg')))
#         frame_hr = np.transpose(frame_hr,(2, 0, 1))/255.0

#         #print(self.hr_dir, video_folder, f"{hr_idx:07d}"+'.jpg')
#         frame_gt = np.array(Image.open(os.path.join(self.hr_dir, frame + '.jpg')))  #load ground truth
#         frame_gt = np.transpose(frame_gt,(2, 0, 1))/255.0


#         return torch.from_numpy(frames), torch.from_numpy(frame_hr), torch.from_numpy(frame_gt), frame

#     def __len__(self):
#         return len(self.pathlist)
    
#     def loadpath(self, pathlistfile):
#         fp = open(pathlistfile)
#         pathlist = fp.read().splitlines()
#         fp.close()
#         return pathlist




# import time
# tic = time.time()
# total = time.time()

# # # # # # Testing_functions 

# # # # # # # # LRtoHR_loader

# # # # train_list = '/home/hmahmad/Documents/mydata/my_toFlow/train_list.csv'
# # # # root = '/home/hmahmad/Documents/mydata/VizDrone2019/'
# # # # data_set = LRtoHR_loader(root , train_list)

# # # # data_loader = DataLoader(data_set,batch_size=8, num_workers=8, shuffle=True)

# # # # for i, (lr,gt,frame_name) in enumerate(data_loader):
# # # #     print(i, lr.shape, gt.shape, lr[0].min(), lr[0].max(), lr.dtype)
# # # #     show_figure(gt[1,:,:,:], lr[1,1,:,:,:], frame_name[1])

# # # # # # # # LRtoHR_with_HR_Loader

# # # train_list = './small_train_list.csv'
# # # root = '/home/mughees/thinclient_drives/VIZDRONE/upsampled/tiny/'
# # # data_set = LRtoHR_with_HR_Loader(root , train_list)

# # # data_loader = DataLoader(data_set,batch_size=1, num_workers=4)#, shuffle=True)

# # # for i, (lr,gt,frame_name) in enumerate(data_loader):
# # #     #print(i, lr.shape, hr.shape, gt.shape, lr[0].min(), lr[0].max())
# # #     CURRENT = time.time()
# # #     print("time took for iter {} is {}".format(i,CURRENT - tic))
# # #     tic = CURRENT

# # # # # # # # LRtoHR_with_HR_Loader_new_terminology

# train_list = '/home/mughees/thinclient_drives/VIZDRONE/upsampled/original/tiny_train_list.csv'
# root = '/home/mughees/thinclient_drives/VIZDRONE/upsampled/original/train/'
# data_set = LRtoHR_with_HR_Loader_new_terminology(root , train_list)

# data_loader = DataLoader(data_set,batch_size=16, num_workers=4)#, shuffle=True)

# for i, (lr,gt,frame_name) in enumerate(data_loader):
#     print(i, lr.shape, gt.shape, lr[0].min(), lr[0].max())
#     CURRENT = time.time()
#     print("time took for iter {} is {}".format(i,CURRENT - tic))
#     tic = CURRENT

# # toc = time.time() -total

# # print('sec:', toc)
import os
import datetime
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import time
from helper import show_figure, show_time, rmse, ssim
from helper import PSNR_function as PSNRatio
import numpy as np
from tqdm import tqdm
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:1" if use_cuda else "cpu")
import torch.nn.functional as F

# ## for LR_HR_HR_HR_interpolation'
# model_name = 'LR_HR_HR_HR_interpolation'
# from Network_HRincluded_new_terminology import TOFlow_LR_HR_inter as TOFlow
# from read_data import LRtoHR_with_HR_Loader_new_terminology as custom_loader

# # ## for HR_HR_interpolation
# model_name = 'HR_HR_interpolation'
# from Network_HRincluded_new_terminology import TOFlow_interpolation as TOFlow
# from read_data import LRtoHR_with_HR_HR_interpolation as custom_loader

# model_name = 'HR_included_tiny' #-> TODO at the end its the network in which we used of LRs for HRs 
# from Network_HRincluded_new_terminology import TOFlow
# from read_data import LRto:HR_with_HR_Loader_new_terminology as custom_loader

# model_name = 'HR_with_low_fps'
# from Network_HRincluded_new_terminology import TOFlow
# from read_data import LRtoHR_with_HR_low_fps_Loader as custom_loader

# model_name = 'HR_with_low_fps_with_random_frame'
# from Network_HRincluded_new_terminology import TOFlow
# from read_data import LRtoHR_with_HR_low_fps_random_frame_Loader as custom_loader

## Base Model
#from Network_HRincluded_new_terminology import TOFlow_base as TOFlow
#from read_data import LRtoHR_loader as custom_loader
#model_name = 'LR_to_HR_base_model'

 
from vizdrone_dataset import HR_HR_interpolation as custom_loader

working_path = '/home/mughees/thinclient_drives/VIZDRONE/upsampled/'

from train_log.RIFE_HDv3 import Model
TOFlow = Model()
TOFlow.load_model("./train_log", -1)
TOFlow.eval()
TOFlow.device()

BATCH_SIZE = 1

model_name = 'testing_interpolation_on_rife'

root = working_path + 'original/'
#just for Low_fps
val_list = root + 'tiny_val_list_7_frames.csv'


def create_dataloader(root):
    val_dataset = custom_loader(root+'val/', val_list)
    val_loader = DataLoader(val_dataset,batch_size=BATCH_SIZE, num_workers=24, shuffle = False)
    print("Dataset Loaded....")
    print("val_loader", len(val_loader)*BATCH_SIZE)
    print("val iterations in each training epoch", len(val_loader))
    return val_loader

h, w = 380, 672
ph = ((h - 1) // 32 + 1) * 32
pw = ((w - 1) // 32 + 1) * 32
padding = (0, pw - w, 0, ph - h)

 
def validation(model, test_, criterion, losses=None, MSE=None, SSIM=None, PSNR=None):
    model.eval()
    #tot_acc=[]
    test_iter=0
    mse = 0;out_ssim =0; t_losses = 0; psnr = 0
    with torch.no_grad():
        for batch_idx, (data, target, path_code) in tqdm(enumerate(test_)):
            data, gt = Variable(data.to(device)), Variable(target.to(device))
            gt = F.pad(gt, padding)
            img1 = F.pad(data[:,0,:,:], padding)
            img2 = F.pad(data[:,1,:,:], padding)
            output = model.inference(img1,img2)
            loss = criterion(output, gt)
            t_losses += loss.item()
            
            pred = output.data.cpu()
            gt = gt.data.cpu()#.float()
            #if batch_idx % 10 == 9:
            #show_figure(gt[0,:,:,:], pred[0,:,:,:], path_code[0], model_name)

            mse_dict = rmse(pred,gt)
            mse += mse_dict['MSE']
            out_ssim += ssim(pred,gt, size_average=True )
            psnr += PSNRatio(mse_dict['RMSE'])
        losses.append(t_losses/len(test_))
        MSE.append(mse/len(test_))
        SSIM.append(out_ssim/len(test_))
        PSNR.append(psnr/len(test_))
        return {"losses":losses,
                "MSE": MSE,
                "RMSE": np.sqrt(MSE),
                "SSIM": SSIM,
                'PSNR': PSNR}


from torch.autograd import Variable
     
def main():
    val_loader = create_dataloader(root = root)
    loss_func = torch.nn.L1Loss()
    
    ## for HR_addition
    # pretrained_dict = torch.load('./toflow_models/myflow_44_best_params.pkl')
    # conv_weights = pretrained_dict['ResNet.conv_3x7_64_9x9.weight'] #64,3,7,7
    # b4 = conv_weights[:,9:12,:,:]#.unsqueeze(1) #64,1,7,7
    # concat_weights = torch.cat([conv_weights,b4],dim = 1) #64,4,7,7 
    # pretrained_dict['ResNet.conv_3x7_64_9x9.weight'] = concat_weights   
    # net.load_state_dict(pretrained_dict) 
    
    #net.load_state_dict(torch.load('./toflow_models/Upsampled_tinyepoch 28.pkl'))  
    #net = train(net, train_loader, val_loader, loss_func, optimizer, epochs = EPOCHS, scheduler=scheduler)
    #net.load_state_dict(torch.load('./toflow_models/retrainingepoch 44_best_params.pkl'))
    #net.load_state_dict(torch.load('./toflow_models/new_terminology_tinyepoch 42.pkl'))
    
    results = validation(TOFlow, val_loader, loss_func, losses=[], MSE=[], SSIM=[], PSNR=[])
    print('SSIM: ',results['SSIM'])
    print('RMSE: ',results['RMSE'])
    print('val_loss: ',results['losses'])
    print('PSNR_if_calcullated at the end: ',PSNRatio(results['RMSE']))
    print('PSNR averaged: ',results['PSNR'])

if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    torch.cuda.manual_seed(0)
    print("pytorch version:", torch.__version__)
    
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    

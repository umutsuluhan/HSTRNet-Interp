import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from queue import Queue
import _thread
import matplotlib.pyplot as plt
from torch.nn import functional as F
from HSTR import Model

def pad_image(img):   
    img = torch.from_numpy(np.transpose(img, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
    return F.pad(img, padding)

def image_input():
    first_video = './images/2X_vid'
    second_video = './images/original_vid'
    
    videogen = []
    for f in os.listdir(first_video):
        if 'png' in f:
            videogen.append(f)
            videogen.sort(key= lambda x:int(x[:-4]))

    first_video_tot_frame = len(videogen)
    first_video_frames = []
    for i in range(videogen.__len__()):
        first_video_frames.append(cv2.imread(os.path.join(first_video, videogen[i]), cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy())

    videogen = []
    for f in os.listdir(second_video):
        if 'png' in f:
            videogen.append(f)
            videogen.sort(key= lambda x:int(x[:-4]))

    second_video_tot_frame = len(videogen)
    second_video_frames = []
    for i in range(videogen.__len__()):
        second_video_frames.append(cv2.imread(os.path.join(second_video, videogen[i]), cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy())    

    frame_mult = (round(first_video_tot_frame / second_video_tot_frame))

    return first_video_frames, second_video_frames, frame_mult 

# for frame in frames:
#     plt.imshow(frame)
#     plt.show()
    

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model()
    #model.eval()
    #model.to(device)
    
    first_video, second_video, frame_mult = image_input()
    
    print(frame_mult)
    
    h, w, _ = first_video[0].shape
    tmp = 32
    ph = ((h - 1) // tmp + 1) * tmp
    pw = ((w - 1) // tmp + 1) * tmp
    padding = (0, pw - w, 0, ph - h)
    
    for i, img in enumerate(first_video):
        first_video[i] = pad_image(first_video[i])
        
    for i, img in enumerate(second_video):
        second_video[i] = pad_image(second_video[i])
    
    
    #for i in range(len(second_video) - 1):
        
    
    imgs = torch.cat((first_video[0], first_video[1], first_video[2], second_video[0], second_video[1]), 1)
    model.inference(imgs)




# while True:
#     frame = read_buffer.get()
#     if frame is None:
#         break
#     I0 = I1
#     I1 = torch.from_numpy(np.transpose(frame, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
#     I1 = pad_image(I1)
#     I0_small = F.interpolate(I0, (32, 32), mode='bilinear', align_corners=False)
#     I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
#     ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])

#     if ssim > 0.995:
#         if skip_frame % 100 == 0:
#             print("\nWarning: Your video has {} static frames, skipping them may change the duration of the generated video.".format(skip_frame))
#         skip_frame += 1
#         if args.skip:
#             pbar.update(1)
#             continue

#     if ssim < 0.2:
#         output = []
#         for i in range((2 ** args.exp) - 1):
#             output.append(I0)
#         '''
#         output = []
#         step = 1 / (2 ** args.exp)
#         alpha = 0
#         for i in range((2 ** args.exp) - 1):
#             alpha += step
#             beta = 1-alpha
#             output.append(torch.from_numpy(np.transpose((cv2.addWeighted(frame[:, :, ::-1], alpha, lastframe[:, :, ::-1], beta, 0)[:, :, ::-1].copy()), (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.)
#         '''
#     else:
#         output = make_inference(I0, I1, 2**args.exp-1) if args.exp else []

# def make_inference(I0, I1, n):
#     global model
#     middle = model.inference(I0, I1, args.scale)
#     if n == 1:
#         return [middle]
#     first_half = make_inference(I0, middle, n=n//2)
#     second_half = make_inference(middle, I1, n=n//2)
#     if n%2:
#         return [*first_half, middle, *second_half]
#     else:
#         return [*first_half, *second_half]





    
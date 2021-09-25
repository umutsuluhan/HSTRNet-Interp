import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from HSTR_FSS import Model
import time

def getting_input(video):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)   
    tot_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    #Getting timestamp of each frames
    timestamps = [cap.get(cv2.CAP_PROP_POS_MSEC)]
    proc_timestamps = [0.0]
    frames = []
    while(cap.isOpened()):
        frame_exists, curr_frame = cap.read()
        if(frame_exists):
            timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
            proc_timestamps.append(timestamps[-1] + 1000/fps)
            frames.append(curr_frame)
        else:
             break
            
    cap.release()
    
    return frames, proc_timestamps, tot_frame, fps

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    

    lfr_video = "videos/car_vid.mp4"             # Low frame rate video
    hfr_video = "videos/car_vid_2X_12fps.mp4"    # High frame rate video

    hstr_model = Model()                      # Initializing the model

    # Processing input videos
    lfr_frames, lfr_proc_timestamps, lfr_tot_frame, lfr_fps = getting_input(lfr_video)  
    hfr_frames, hfr_proc_timestamps, hfr_tot_frame, hfr_fps = getting_input(hfr_video)
    
    # Concatenating input images into one to feed to the network
    imgs = np.concatenate((lfr_frames[0], lfr_frames[1], hfr_frames[0], hfr_frames[1], hfr_frames[2]), 2)
    
    # Feeding images to the model
    hstr_model.inference(imgs, hfr_proc_timestamps[:3])

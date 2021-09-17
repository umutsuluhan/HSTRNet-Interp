import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from HSTR import Model
import time

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    

    lfr_video = "videos/demo.mp4"
    hfr_video = "videos/demo_2X_50fps.mp4"

    hstr_model = Model()

    start_time = time.time()

    #Getting fps and total frame number of low frame rate video
    lfr_video_capture = cv2.VideoCapture(lfr_video)
    lfr_fps = lfr_video_capture.get(cv2.CAP_PROP_FPS)   
    lfr_tot_frame = lfr_video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    
    #Getting timestamp of each frames
    lfr_timestamps = [lfr_video_capture.get(cv2.CAP_PROP_POS_MSEC)]
    lfr_proc_timestamps = []
    lfr_frames = []
    while(lfr_video_capture.isOpened()):
        frame_exists, curr_frame = lfr_video_capture.read()
        if(frame_exists):
            lfr_timestamps.append(lfr_video_capture.get(cv2.CAP_PROP_POS_MSEC))
            lfr_proc_timestamps.append(lfr_timestamps[-1] + 1000/lfr_fps)
            lfr_frames.append(curr_frame)
        else:
             break
            
    lfr_video_capture.release()
    
    #Getting fps and total frame number of high frame rate video
    hfr_video_capture = cv2.VideoCapture(hfr_video)
    hfr_fps = hfr_video_capture.get(cv2.CAP_PROP_FPS)
    hfr_tot_frame = hfr_video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    
    #Getting timestamp of each frames
    hfr_timestamps = [hfr_video_capture.get(cv2.CAP_PROP_POS_MSEC)]
    hfr_proc_timestamps = []
    hfr_frames = []
    while(hfr_video_capture.isOpened()):
        frame_exists, curr_frame = hfr_video_capture.read()
        if(frame_exists):
            hfr_timestamps.append(hfr_video_capture.get(cv2.CAP_PROP_POS_MSEC))
            hfr_proc_timestamps.append(hfr_timestamps[-1] + 1000/lfr_fps)
            hfr_frames.append(curr_frame)
        else:
            break
    
    hfr_video_capture.release()
   
    finish_time = time.time()

    imgs = np.concatenate((lfr_frames[0], lfr_frames[1], hfr_frames[0], hfr_frames[1], hfr_frames[2]), 2)
    timestamps = []
    timestamps.append(lfr_proc_timestamps[0])
    timestamps.append(lfr_proc_timestamps[1])
    timestamps.append(hfr_proc_timestamps[0])
    timestamps.append(hfr_proc_timestamps[1])
    timestamps.append(hfr_proc_timestamps[2])
    
    hstr_model.inference(imgs, timestamps)

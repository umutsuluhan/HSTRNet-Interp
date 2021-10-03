import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from model.FastRIFE_Super_Slomo.HSTR_FSS import HSTR_FSS
from model.FastRIFE_Super_Slomo.HSTR_LKSS import HSTR_LKSS



def getting_input(video):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    tot_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # Getting timestamp of each frames
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

    hr_video = "videos/car_vid.mp4"             # Low frame rate video
    lr_video = "videos/car_vid_2X_12fps.mp4"    # High frame rate video

    hstr_model = HSTR_FSS()                      # Initializing the model
    # HSTR_FSS.to(device)

    # Processing input videos
    hr_frames, hr_proc_timestamps, hr_tot_frame, hr_fps = getting_input(
        hr_video)
    lr_frames, lr_proc_timestamps, lr_tot_frame, lr_fps = getting_input(
        lr_video)

    # Concatenating input images into one to feed to the network
    #imgs = np.concatenate((lfr_frames[0], lfr_frames[1], hfr_frames[0], hfr_frames[1], hfr_frames[2]), 2)

    # Feeding images to the model
    output = []
    for i in range(len(hr_frames) - 1):
        imgs = np.concatenate(
            (hr_frames[i], hr_frames[i + 1], lr_frames[2*i], lr_frames[2*i + 1], lr_frames[2*i + 2]), 2)
        output.append(hr_frames[i])
        output_image = hstr_model.inference(imgs, lr_proc_timestamps[:3])
        output.append(output_image)

    height, width, layers = output[0].shape
    size = (width, height)
    four_cc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter('project.mp4', four_cc, 12, size)

    # print(len(output))
    for i in range(len(output)):
        cv2.waitKey(100)
        if str(output[i].dtype) != "uint8":
            temp = output[i]
            output[i] = cv2.normalize(src=temp, dst=None, alpha=0, beta=255,
                                      norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            # cv2.imshow("win", output[i])
            # cv2.waitKey(100)
        out.write(output[i])
    out.release()

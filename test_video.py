import cv2
import matplotlib.pyplot as plt

if __name__ == '__main__':

    lfr_video = "videos/demo.mp4"
    hfr_video = "videos/demo_2X_50fps.mp4"

    lfr_video_capture = cv2.VideoCapture(lfr_video)
    lfr_fps = lfr_video_capture.get(cv2.CAP_PROP_FPS)
    lfr_tot_frame = lfr_video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    
    lfr_timestamps = [lfr_video_capture.get(cv2.CAP_PROP_POS_MSEC)]
    lfr_proc_timestamps = [0.0]
    while(lfr_video_capture.isOpened()):
        frame_exists, curr_frame = lfr_video_capture.read()
        if(frame_exists):
            plt.imshow(curr_frame)
        #     lfr_timestamps.append(lfr_video_capture.get(cv2.CAP_PROP_POS_MSEC))
        #     frame_and_timestamp = (curr_frame, lfr_timestamps[-1] + 1000/lfr_fps)
        #     lfr_proc_timestamps.append(curr_frame : lfr_timestamps[-1] + 1000/lfr_fps)
        else:
             break
            
    lfr_video_capture.release()
    
    hfr_video_capture = cv2.VideoCapture(hfr_video)
    hfr_fps = hfr_video_capture.get(cv2.CAP_PROP_FPS)
    hfr_tot_frame = hfr_video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    hfr_timestamps = [hfr_video_capture.get(cv2.CAP_PROP_POS_MSEC)]
    
    
    hfr_timestamps = [hfr_video_capture.get(cv2.CAP_PROP_POS_MSEC)]
    hfr_proc_timestamps = [0.0]
    while(hfr_video_capture.isOpened()):
        frame_exists, curr_frame = hfr_video_capture.read()
        if(frame_exists):
            hfr_timestamps.append(hfr_video_capture.get(cv2.CAP_PROP_POS_MSEC))
            hfr_proc_timestamps.append(hfr_timestamps[-1] + 1000/lfr_fps)
        else:
            break
    
    hfr_video_capture.release()

    
    print(len(lfr_proc_timestamps))
    print(len(hfr_timestamps))
    print(lfr_proc_timestamps[0])

    # videogen = skvideo.io.vreader(args.video)
    # lastframe = next(videogen)
    # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # video_path_wo_ext, ext = os.path.splitext(args.video)
    # print('{}.{}, {} frames in total, {}FPS to {}FPS'.format(
    # video_path_wo_ext, args.ext, tot_frame, fps, args.fps))

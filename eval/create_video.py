import cv2
import numpy as np
import glob

img_array = []
for filename in sorted(glob.glob('/home/mughees/thinclient_drives/VIZDRONE/upsampled/original/val/LR/uav0000086_00000_v/*.jpg')):
    #f = filename.split("/")[10].split(".")[0]
    #if int(f) % 2 == 1:
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('LR.mp4', fourcc, 24, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
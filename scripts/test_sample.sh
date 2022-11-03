CUDA_VISIBLE_DEVICES=1 python3 eval/test_sample.py --image1_HR /home/mughees/thinclient_drives/VIZDRONE/upsampled/original/val/HR/uav0000086_00000_v/0000001.jpg \
               --image2_HR /home/mughees/thinclient_drives/VIZDRONE/upsampled/original/val/HR/uav0000086_00000_v/0000003.jpg \
               --image1_LR /home/mughees/thinclient_drives/VIZDRONE/upsampled/original/val/LR/uav0000086_00000_v/0000001.jpg \
               --image2_LR /home/mughees/thinclient_drives/VIZDRONE/upsampled/original/val/LR/uav0000086_00000_v/0000002.jpg \
               --image3_LR /home/mughees/thinclient_drives/VIZDRONE/upsampled/original/val/LR/uav0000086_00000_v/0000003.jpg \
               --warping 1 \
               --model_path ./pretrained/warp

# Leave warping 0 if you want to experiment deformable convolution model, if warp then change to 1
# Set model path deformable convolution for warping == 0, else set it is as warp
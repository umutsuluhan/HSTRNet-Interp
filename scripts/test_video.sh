CUDA_VISIBLE_DEVICES=2 python3 eval/test_video.py --HR_video ./sample/videos/HR_12fps.mp4 \
                --LR_video ./sample/videos/LR_24fps.mp4 \
                --warping 0 \
                --model_path ./pretrained/deformable_convolution/

# Leave warping 0 if you want to experiment deformable convolution model, if warp then change to 1
# Set model path deformable convolution for warping == 0, else set it is as warp
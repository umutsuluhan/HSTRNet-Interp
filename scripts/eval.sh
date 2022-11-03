CUDA_VISIBLE_DEVICES=1 python3 eval/eval.py --dataset_name Vimeo \
               --dataset_path /home/ortak/mughees/datasets/vimeo_triplet \
               --warping 1 \
               --model_path ./pretrained/warp/ 

# Leave warping 0 if you want to experiment deformable convolution model, if warp then change to 1
# Set model path deformable_convolution for warping == 0, else set it is as warp

               
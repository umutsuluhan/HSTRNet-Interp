CUDA_VISIBLE_DEVICES=1 python3 train.py --data_root /home/ortak/mughees/datasets/vimeo_triplet \
               --train_batch_size 64 \
               --val_batch_size 16 \
               --epoch 100 \
               --checkpoint 0 \
               --checkpoint_start 45 \
               --checkpoint_path ./checkpoint \
               --warping 1 /
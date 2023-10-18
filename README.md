# HSTRNet-interp: Dual Camera Based High Spatio-Temporal Resolution Video Generation For Wide Area Surveillance
## Introduction

This repository contains implementation of paper titled *Dual Camera Based High Spatio-Temporal Resolution Video Generation For Wide Area Surveillance*([IEEE](https://ieeexplore.ieee.org/abstract/document/9959711/)). HSTRNet-interp is a deep-learning network capable of generating high spatiotemporal resolution videos from high spatial resolution - low frame rate and low spatial resolution - high frame rate feeds.


## Getting Started

Run following commands to setup the environment and download weights to proper locations:
```
git clone https://github.com/umutsuluhan/HSTRNet-Interp.git
source create_env.sh
```

## Testing

Run following command to test model on different datasets:
```
python3 scripts/eval.py --dataset_name <dataset_name> --dataset_path <dataset_path> --model_path <model_path>
```

Parameters are:

| dataset_name | Vimeo | Vizdrone |
| ------ | ------ | ------|
| dataset_path | Vimeo dataset path | pretrained/vimeo |
| dataset_name | Visdrone dataset path | pretrained/visdrone |

## Training

Run following command to train the model:
```
python3 scripts/train.py --data_root <dataset_path> --train_batch_size <training_batch_size> --val_batch_size <validation_batch_size> --epoch <epoch_count> --checkpoint <0,1> --checkpoint_start <checkpoint_epoch_number> --checkpoint_path <checkpoint_path>
```

- checkpoint flag is set to 0 if training is being started from scratch and 1 if training will continue from a checkpoint
- checkpoint_start is the epoch number to continue if checkpoint is set to 1

## Citation

Please cite our paper in the following format if you use this codebase for academic purposes:
```
@inproceedings{suluhan2022dual,
  title={Dual Camera Based High Spatio-Temporal Resolution Video Generation For Wide Area Surveillance},
  author={Suluhan, H Umut and Ates, Hasan F and Gunturk, Bahadir K},
  booktitle={2022 18th IEEE International Conference on Advanced Video and Signal Based Surveillance (AVSS)},
  pages={1--8},
  year={2022},
  organization={IEEE}
}
```
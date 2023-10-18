#!/bin/bash


#python -m venv release	
#source $PWD/release/bin/activate

#Dependencies
#pip install --upgrade pip
#pip install torch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0
#pip install mmcv==2.0.0rc4 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.12/index.html
#pip install tqdm
#pip install gdown

mkdir pretrained
mkdir pretrained/vimeo
mkdir pretrained/visdrone

gdown 1REeqQlRyYcS7Zkx_TPoVmL5R3EdUiHMn
gdown 1GeLtmFkyuZWsyHl4MqLhJSDKQGGLP1Sb
gdown 1Y-wPS2m6MoHjFxdYxgbEFxeowMHW5S3Y
mv ifnet.pkl contextnet.pkl unet.pkl pretrained/vimeo/

gdown 1atm7KEqx0Az5IlMV-P_Iu5sBoR7HL68D
gdown 1PNbHOx3vMi2hsJGRx-citWTTLS204g-j
gdown 1bTKQmoaTSBnYOsYT8JMAtTPN44CG3Ghz
mv ifnet.pkl contextnet.pkl unet.pkl pretrained/visdrone/


# HSTRNet

Deep neural network model to synthesize high resolution-high frame rate video from low resolution-high frame rate and high_resolution-low frame rate videos.

Deep neural network model to synthesize high resolution-high frame rate video from low resolution-high frame rate and high_resolution-low frame rate videos.  

original_models --> Kullanılan modellerin orijinal halleri  

HSTR_Farneback_SuperSlomo --> Benim yazdığım model  

HSTR_Farneback_SuperSlomo/dataset.py --> Dataset kodu  
HSTR_Farneback_SuperSlomo/test_video.py --> Video test kodu  
HSTR_Farneback_SuperSlomo/train_HSTR_FSS.py --> Training kodu  

HSTR_Farneback_SuperSlomo/model/HSTR_FSS.py --> Farneback + SuperSlomo kodu  
HSTR_Farneback_SuperSlomo/model/HSTR_LKSS.py --> Lucas Kanade + SuperSlomo kodu (Farneback kadar güncel değil)  
HSTR_Farneback_SuperSlomo/model/backwarp.py --> Backwarping modülü  
HSTR_Farneback_SuperSlomo/model/unet_model.py --> Unet modülü  
HSTR_Farneback_SuperSlomo/model/unet_parts.py --> Unet helper modülü  

RIFE_HSTR --> RIFE'a 5 input verilen model  

RIFE_HSTR/dataset.py --> Dataset kodu  
RIFE_HSTR/inference_video.py --> Video test kodu  
RIFE_HSTR/train_RIFE_HSTR.py --> Training kodu (Evaluation kısmı daha bitmedi)  

RIFE_HSTR/RIFE/RIFE_scaled.py --> 5 input RIFE kodu 2 X scaled  
RIFE_HSTR/RIFE/RIFE_HSTR_Lucas_Kanade.py --> LR için Lucas Kanade
RIFE_HSTR/RIFE/RIFE_HSTR_Farneback.py --> LR için Farneback
RIFE_HSTR/RIFE/IFNet.py --> Optical flow kodu (normal kod ile birebir aynı)  
RIFE_HSTR/RIFE/loss.py --> Training'de kullanılan loss functionlar  
RIFE_HSTR/RIFE/warplayer.py --> Backwarping modülü  
RIFE_HSTR/RIFE/pytorch_msssim --> Ssim modülü  
RIFE_HSTR/RIFE/other_models --> Farklı RIFE modelleri  

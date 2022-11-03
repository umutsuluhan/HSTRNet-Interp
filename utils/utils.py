import numpy as np
import cv2
import math
import torch
import torch.nn.functional as F
import torch.nn as nn

device = "cuda"

def image_show(result):
    result = result.cpu().detach().numpy()
    result = result[0, :]
    result = np.transpose(result, (1, 2, 0))
    cv2.imshow("win", result)
    cv2.waitKey(2000)

def crop(pred, gt, w, h):
    _, _ , iw, ih = pred.shape
    x = int((iw - w) / 2)
    y = int((ih - h) / 2)
    pred = pred[:, :, x:iw-x, y:ih-y]
    gt = gt[:, :, x:iw-x, y:ih-y]
    return pred, gt

def crop_inference(pred, w, h):
    _, _ , iw, ih = pred.shape
    x = int((iw - w) / 2)
    y = int((ih - h) / 2)
    pred = pred[:, :, x:iw-x, y:ih-y]
    return pred

def convert(param):
    return {k.replace("module.", ""): v for k, v in param.items() if "module." in k}

def convert_to_torch(img):
    img = torch.from_numpy(img.copy()).permute(2, 0, 1).unsqueeze(0).to(device).float() / 255.0
    return img

def convert_to_numpy(img):
    result = img.cpu().detach().numpy()
    result = result[0, :]
    result = np.transpose(result, (1, 2, 0))
    return result * 255

def down_up(img):
    result = F.interpolate(img, scale_factor=0.25, mode="bicubic", align_corners=False)
    result = F.interpolate(result, scale_factor=4, mode="bicubic", align_corners=False)
    result = result.cpu().detach().numpy()
    result = result[0, :]
    result = np.transpose(result, (1, 2, 0))
    frame_normed = 255 * result
    frame_normed = np.array(frame_normed, np.int)
    return frame_normed

def padding_vis(img):
    if img.shape[2] % 64 != 0:
        div = math.floor(img.shape[2] / 32)
        padding1_mult = div + 1 if div % 2 == 1 else div + 2
    else:
        padding1_mult = math.floor(img.shape[2] / 32)

    if img.shape[3] % 64 != 0:
        div = math.floor(img.shape[3] / 32)
        padding2_mult = div + 1 if div % 2 == 1 else div + 2
    else:
        padding2_mult = math.floor(img.shape[3] / 32)


    pad1 = (32 * padding1_mult) - img.shape[2]
    pad2 = (32 * padding2_mult) - img.shape[3]

    img = torch.unsqueeze(img, 0)
    padding = nn.ZeroPad2d((int(pad2/2), int(pad2/2), int(pad1/2), int(pad1/2)))
    img = img.float()
    img = padding(img)
    img = torch.squeeze(img, 0)
    return img

def calc_psnr(img0, img1):
    return -10 * math.log10(((img0 - img1) * (img0 - img1)).mean())

def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window_3d(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _2D_window.unsqueeze(2) @ (_1D_window.t())
    window = _3D_window.expand(1, channel, window_size, window_size, window_size).contiguous().to(device)
    return window

def ssim_matlab(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, _, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window_3d(real_size, channel=1).to(img1.device)
        # Channel is set to 1 since we consider color images as volumetric images

    img1 = img1.unsqueeze(1)
    img2 = img2.unsqueeze(1)

    mu1 = F.conv3d(F.pad(img1, (5, 5, 5, 5, 5, 5), mode='replicate'), window, padding=padd, groups=1)
    mu2 = F.conv3d(F.pad(img2, (5, 5, 5, 5, 5, 5), mode='replicate'), window, padding=padd, groups=1)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv3d(F.pad(img1 * img1, (5, 5, 5, 5, 5, 5), 'replicate'), window, padding=padd, groups=1) - mu1_sq
    sigma2_sq = F.conv3d(F.pad(img2 * img2, (5, 5, 5, 5, 5, 5), 'replicate'), window, padding=padd, groups=1) - mu2_sq
    sigma12 = F.conv3d(F.pad(img1 * img2, (5, 5, 5, 5, 5, 5), 'replicate'), window, padding=padd, groups=1) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
from math import exp
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import cv2
from torchvision.transforms import Compose,ToTensor,ToPILImage
from scipy.fftpack import fft
from math import log10


"read .hd5 file and convert to ndarray"
def read_h5(path):
    f = h5py.File(path, "r")
    data = f["/data"]
    return np.array(data)


"""save ndarray to .hd5 file"""
def save_h5(data, path):
    f = h5py.File(path, "w")
    f.create_dataset("/data", data=data)
    f.close()

def gaussian(window_size,sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size,channel):
    _1D_window = gaussian(window_size,1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1,img2,window,window_size,channel,size_average = True):
    mu1 = F.conv2d(img1,window,padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    if not torch.is_tensor(img1):
        to_tensor = ToTensor()
        img1 = to_tensor(img1)
    if not torch.is_tensor(img2):
        to_tensor = ToTensor()
        img2 = to_tensor(img2)
    if img1.dim() == 3:
        img1 = torch.unsqueeze(img1,dim = 0)
    elif img1.dim() == 2:
        img1 = torch.unsqueeze(torch.unsqueeze(img1, dim=0),dim=0)
    if img2.dim() == 3:
        img2 = torch.unsqueeze(img2, dim=0)
    elif img2.dim() == 2:
        img2 = torch.unsqueeze(torch.unsqueeze(img2, dim=0), dim=0)
    img1 = (img1 - torch.min(img1)) / (torch.max(img1) - torch.min(img1))
    img2 = (img2 - torch.min(img2)) / (torch.max(img2) - torch.min(img2))
    try:
        (_, channel, _, _) = img1.size()
    except:
        channel = 1
    window = create_window(window_size, channel)
    try:
        if img1.is_cuda:
            window = window.cuda(img1.get_device())
        window = window.type_as(img1)
    except:
        window = window.type_as(img1)
    return _ssim(img1, img2, window, window_size, channel, size_average)

def display_transform():
    return Compose([
        ToPILImage(),
        ToTensor()
    ])


def fft_trace(trace,dt,n):
    fs = 1 / dt
    N = np.linspace(0,n-1,n)
    y = fft(trace,n)
    mag = abs(y)
    f = N * fs / n
    amp = mag[:int(n / 2)]
    freq = f[:int(n / 2)]
    return amp,freq


def get_amp(data,dt):
    x,y = data.shape    # x:trace,y:sample number
    amp_2d = np.zeros((x,int(y / 2)))
    for i in range(x):
        amp_2d[i,:],f = fft_trace(trace = data[i,:],dt = dt,n = y)
    return amp_2d

def frequency_distance(data,target,dt):
    target_max = np.max(target)
    target_min = np.min(target)
    data = (target_max - target_min) * data + target_min
    data_amp = get_amp(data,dt)
    target_amp = get_amp(target,dt)
    M,N = data_amp.shape
    distance = np.sqrt(np.sum((data_amp[:,3:] - target_amp[:,3:])**2) / (M * N))
    return distance

def normal(data):
    data_max = np.max(data)
    data_min = np.min(data)
    return (data - data_min) / (data_max - data_min)

def cal_psnr(sr,hr):
    sr = normal(sr)
    hr = normal(hr)
    mse = ((sr - hr) ** 2).mean()
    psnr = 10 * log10(np.max(hr) ** 2 / mse)
    return psnr


if __name__ == "__main__":
    file_name1 = "case83_crossline_15.h5"
    file_name2 = "case90_crossline_3.h5"

    lr_path = r"F:\SRGAN_experiment\data\data\SRF_2\test\low"
    hr_path = r"F:\SRGAN_experiment\data\data\SRF_2\test\high"
    sr_original = r"F:\SRGAN_experiment\SRGAN_result\result5\SRF_2\predcited"
    hr_data1 = read_h5(os.path.join(hr_path, file_name1))
    hr_data2 = read_h5(os.path.join(hr_path, file_name2))

    sr_original_data1 = read_h5(os.path.join(sr_original, file_name1))
    sr_original_data2 = read_h5(os.path.join(sr_original, file_name2))
    psnr_original1 = cal_psnr(sr_original_data1, hr_data1)
    psnr_original2 = cal_psnr(sr_original_data2, hr_data2)
    ssim_original1 = ssim(sr_original_data1, hr_data1).numpy()
    ssim_original2 = ssim(sr_original_data2, hr_data2).numpy()
    fd1 = frequency_distance(sr_original_data1, hr_data1, dt=0.002)
    fd2 = frequency_distance(sr_original_data2, hr_data2, dt=0.002)
    print(psnr_original1, psnr_original2)
    print(ssim_original1, ssim_original2)
    print(fd1, fd2)
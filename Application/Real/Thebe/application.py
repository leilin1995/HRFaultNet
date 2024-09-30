"""
__author__ = 'linlei'
__project__:application
__time__:2022/8/28 10:54
__email__:"919711601@qq.com"
"""
from ImproveResolution.code.model import Generator
from ImproveResolution.code.utils import read_h5, save_h5, normal
from FaultSegmentation.code.model import Unet
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
import time

mpl.rcParams['font.sans-serif'] = ['Times New Roman']  # font type
mpl.rcParams['axes.unicode_minus'] = False


def main():
    UPSCALE_FACTOR = 2
    raw_data = read_h5("./seismic_slice.h5")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # load HRNet
    hr_checkpoint = "../../../ImproveResolution/result/SRF_2/model/netG_bestmodel.pth"
    hr_model = Generator(scale_factor=UPSCALE_FACTOR).eval()
    hr_model.load_state_dict(torch.load(hr_checkpoint, map_location=device))
    # load FaultNet
    fault_checkpoint = "../../../FaultSegmentation/result_systhetic_2D/best_model.pth"
    fault_model = Unet()
    loader = torch.load(fault_checkpoint, map_location=device)
    fault_model.load_state_dict(loader["model_state_dict"])
    with torch.no_grad():
        # 1. normalized raw data and conver to tensor
        raw_data = normal(raw_data).T
        raw_data = torch.from_numpy(raw_data).type(torch.FloatTensor)

        # 开始记录超分辨率用时
        start_time_sr = time.time()

        # 2. get hr data from raw data
        input_hr = torch.unsqueeze(raw_data, dim=0)  # add channel dim
        input_hr = torch.unsqueeze(input_hr, dim=0)  # add batch dim
        input_hr.to(device)
        output_hr = hr_model(input_hr)  # hr data

        # 结束记录超分辨率用时
        end_time_sr = time.time()
        print(f"超分辨率处理时间: {end_time_sr - start_time_sr:.4f} 秒")

        # 开始记录断层识别用时
        start_time_fault = time.time()

        # 3. get hr fault map
        fault_hr = fault_model(output_hr).cpu()
        fault_hr = np.squeeze(fault_hr.numpy(), axis=(0, 1))

        # 结束记录断层识别用时
        end_time_fault = time.time()
        print(f"高分辨率断层识别处理时间: {end_time_fault - start_time_fault:.4f} 秒")
        # 4. save hr data and hr fault map
        save_h5(data=np.squeeze(output_hr.cpu().numpy(), axis=(0, 1)).T, path="./hr_seismic.h5")
        save_h5(data=fault_hr, path="./hr_fault.h5")

        # 5. get raw fault map for compare
        # 开始记录断层识别用时
        start_time_fault = time.time()
        fault_raw = fault_model(input_hr).cpu()
        fault_raw = np.squeeze(fault_raw.numpy(), axis=(0, 1))
        # 结束记录断层识别用时
        end_time_fault = time.time()
        print(f"原始断层识别处理时间: {end_time_fault - start_time_fault:.4f} 秒")
        # 6. save raw fault map
        save_h5(data=fault_raw, path="./raw_fault.h5")


def show_result():
    # 1.load data
    seismic_sr = read_h5("./hr_seismic.h5")
    fault_sr = read_h5("./hr_fault.h5")
    # fault_sr[fault_sr<0.3]=0
    sr_alpha = np.ones_like(fault_sr)
    sr_alpha[fault_sr < 0.55] = 0.5
    sr_alpha[fault_sr <= 0.45] = 0.25
    sr_alpha[sr_alpha <= 0.3] = 0.
    seismic_raw = read_h5("./seismic_slice.h5")
    fault_raw = read_h5("./raw_fault.h5")
    # fault_raw[fault_raw < 0.3] = 0
    raw_alpha = np.ones_like(fault_raw)
    raw_alpha[fault_raw < 0.55] = 0.5
    raw_alpha[fault_raw <= 0.45] = 0.25
    raw_alpha[fault_raw <= 0.3] = 0.
    # renormal seismic_sr
    seismic_sr = normal(seismic_sr)
    seismic_sr = seismic_sr * (np.max(seismic_raw) - np.min(seismic_raw)) + np.min(seismic_raw)
    # print(np.max(seismic_raw),np.min(seismic_raw))

    font1 = {'family': 'Times New Roman',
             'color': 'black',
             'weight': 'normal',
             'size': 16,
             }

    fig = plt.figure(figsize=(9, 10))
    gs = GridSpec(50, 62)

    ax1 = fig.add_subplot(gs[0:20, 0:30])
    ax1.imshow(seismic_raw.T, cmap=plt.cm.gray, vmin=-1.6, vmax=1.65)
    ax1.set_ylabel("Samples", font1)
    ax1.set_title("Traces", font1)
    ax1.text(-12, -14, "(a)", font1)

    ax2 = fig.add_subplot(gs[0:20, 32:62])
    ax2.imshow(seismic_raw.T, cmap=plt.cm.gray, vmin=-1.6, vmax=1.65)
    ax2.imshow(fault_raw, cmap="jet", vmin=0.15, vmax=0.85, alpha=raw_alpha)
    ax2.text(-12, -14, "(b)", font1)
    ax2.set_title("Traces", font1)

    ax3 = fig.add_subplot(gs[24:44, 0:30])
    ax3.imshow(seismic_sr.T, cmap=plt.cm.gray, vmin=-1.6, vmax=1.65)
    ax3.set_ylabel("Samples", font1)
    ax3.text(-24, -28, "(c)", font1)

    ax4 = fig.add_subplot(gs[24:44, 32:62])
    ax4.imshow(seismic_sr.T, cmap=plt.cm.gray, vmin=-1.6, vmax=1.65)
    ax4.imshow(fault_sr, cmap="jet", vmin=0.15, vmax=0.85, alpha=sr_alpha)
    ax4.text(-24, -28, "(d)", font1)

    # define colorbar
    dx1 = fig.add_subplot(gs[46:47, 2:28])
    norm1 = mpl.colors.Normalize(vmin=-1.6, vmax=1.65)
    dbar1 = fig.colorbar(mpl.cm.ScalarMappable(norm=norm1, cmap=plt.cm.gray),
                         ticks=[-1.5, -1, -0.5, 0, 0.5, 1, 1.5],
                         cax=dx1,
                         orientation="horizontal")
    dbar1.set_label("Amplitude", loc='center', **font1)

    dx2 = fig.add_subplot(gs[46:47, 34:60])
    norm2 = mpl.colors.Normalize(vmin=0.15, vmax=0.85)
    dbar2 = fig.colorbar(mpl.cm.ScalarMappable(norm=norm2, cmap="jet"),
                         ticks=[0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85],
                         cax=dx2,
                         orientation="horizontal")
    dbar2.set_label("Fault probability", loc='center', **font1)
    plt.savefig("./compare.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
    # show_result()

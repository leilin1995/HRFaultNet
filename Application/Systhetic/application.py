"""
__author__ = 'linlei'
__project__:application
__time__:2022/8/29 19:15
__email__:"919711601@qq.com"
"""
import os
from ImproveResolution.code.model import Generator
from ImproveResolution.code.utils import read_h5,save_h5,normal
from FaultSegmentation.code.model import Unet
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from mayavi.mlab import pipeline
from mayavi import mlab

mpl.rcParams['font.sans-serif'] = ['Times New Roman']  # font type
mpl.rcParams['axes.unicode_minus'] = False



def predict(raw_data_path,save_path):
    UPSCALE_FACTOR = 2
    raw_data=np.rot90(read_h5(raw_data_path)[:,120,:])  # 120th slice in inlines
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # load HRNet
    hr_checkpoint="../../ImproveResolution/result/SRF_2/model/netG_bestmodel.pth"
    hr_model = Generator(scale_factor=UPSCALE_FACTOR).eval()
    hr_model.load_state_dict(torch.load(hr_checkpoint, map_location=device))
    # load FaultNet
    fault_checkpoint="../../FaultSegmentation/result_systhetic_2D/best_model.pth"
    fault_model=Unet()
    loader = torch.load(fault_checkpoint, map_location=device)
    fault_model.load_state_dict(loader["model_state_dict"])
    with torch.no_grad():
        # 1. normalized raw data and conver to tensor
        raw_data=normal(raw_data)
        raw_data=torch.from_numpy(raw_data).type(torch.FloatTensor)
        # 2. get hr data from raw data
        input=torch.unsqueeze(raw_data,dim=0)    # add channel dim
        input=torch.unsqueeze(input,dim=0)    # add batch dim
        input.to(device)
        # 3. get raw fault map for compare
        fault_raw=fault_model(input).cpu()
        fault_raw = np.squeeze(fault_raw.numpy(), axis=(0, 1))
        # 4. save raw fault map
        save_h5(data=fault_raw,path=save_path)

def main():
    dir = "./seismic"
    files_list=os.listdir(dir)
    save_dir="./fault"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for files in files_list:
        raw_data_path=os.path.join(dir,files)
        save_path=save_dir+"/fault_"+files
        predict(raw_data_path,save_path)

def show_result():
    # 1.load data
    seismic_f_10_snr_3 = np.rot90(read_h5("./seismic/traces_f_10_snr_3.h5")[:,120,:])
    fault_f_10_snr_3 = read_h5("./fault/fault_traces_f_10_snr_3.h5")
    alpha_f_10=np.ones_like(fault_f_10_snr_3)
    alpha_f_10[fault_f_10_snr_3 < 0.55] = 0.5
    alpha_f_10[fault_f_10_snr_3 <= 0.45] = 0.25
    alpha_f_10[fault_f_10_snr_3 <= 0.3] = 0.

    seismic_f_20_snr_6 = np.rot90(read_h5("./seismic/traces_f_20_snr_6.h5")[:,120,:])
    fault_f_20_snr_6 = read_h5("./fault/fault_traces_f_20_snr_6.h5")
    alpha_f_20 = np.ones_like(fault_f_20_snr_6)
    alpha_f_20[fault_f_20_snr_6 < 0.55] = 0.5
    alpha_f_20[fault_f_20_snr_6 <= 0.45] = 0.25
    alpha_f_20[fault_f_20_snr_6 <= 0.3] = 0.

    seismic_f_40_snr_12 = np.rot90(read_h5("./seismic/traces_f_40_snr_12.h5")[:,120,:])
    fault_f_40_snr_12 = read_h5("./fault/fault_traces_f_40_snr_12.h5")
    alpha_f_40 = np.ones_like(fault_f_40_snr_12)
    alpha_f_40[fault_f_40_snr_12 < 0.55] = 0.5
    alpha_f_40[fault_f_40_snr_12 <= 0.45] = 0.25
    alpha_f_40[fault_f_40_snr_12 <= 0.3] = 0.

    seismic_f_80 = np.rot90(read_h5("./seismic/traces_f_80.h5")[:,120,:])
    fault_f_80 = read_h5("./fault/fault_traces_f_80.h5")
    alpha_f_80 = np.ones_like(fault_f_80)
    alpha_f_80[fault_f_80 < 0.55] = 0.5
    alpha_f_80[fault_f_80 <= 0.45] = 0.25
    alpha_f_80[fault_f_80 <= 0.3] = 0.
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(50, 62)
    font = {'family': 'Times New Roman',
            'color': 'black',
            'weight': 'normal',
            'size': 16,
            }

    font1 = {'family': 'Times New Roman',
             'color': 'black',
             'weight': 'normal',
             'size': 16,
             }

    ax1 = fig.add_subplot(gs[0:20, 0:30])
    ax1.imshow(seismic_f_10_snr_3, cmap=plt.cm.gray)
    ax1.imshow(fault_f_10_snr_3,vmin=0,vmax=1,cmap="jet",alpha=alpha_f_10)
    ax1.set_ylabel("Samples", font)
    ax1.set_title("Traces", font)
    ax1.text(-12, -14, "(a)", font1)

    ax2 = fig.add_subplot(gs[0:20, 32:62])
    ax2.imshow(seismic_f_20_snr_6, cmap=plt.cm.gray)
    ax2.imshow(fault_f_20_snr_6,vmin=0,vmax=1,cmap="jet",alpha=alpha_f_20)
    ax2.text(-12, -14, "(b)", font1)
    ax2.set_title("Traces", font)

    ax3 = fig.add_subplot(gs[24:44, 0:30])
    ax3.imshow(seismic_f_40_snr_12, cmap=plt.cm.gray)
    ax3.imshow(fault_f_40_snr_12,vmin=0,vmax=1,cmap="jet",alpha=alpha_f_40)
    ax3.set_ylabel("Samples", font)
    ax3.text(-24, -28, "(c)", font1)

    ax4 = fig.add_subplot(gs[24:44, 32:62])
    ax4.imshow(seismic_f_80, cmap=plt.cm.gray)
    ax4.imshow(fault_f_80,vmin=0,vmax=1,cmap="jet",alpha=alpha_f_80)
    ax4.text(-24, -28, "(d)", font1)

    # define colorbar
    dx1 = fig.add_subplot(gs[46:47, 0:30])
    norm1 = mpl.colors.Normalize(vmin=-7.5, vmax=7.5)
    dbar1 = fig.colorbar(mpl.cm.ScalarMappable(norm=norm1, cmap=plt.cm.gray),
                         ticks=[-7.5, -5, -2.5, 0, 2.5, 5, 7.5],
                         cax=dx1,
                         orientation="horizontal")
    dbar1.set_label("Amplitude", loc='center', **font1)

    dx2 = fig.add_subplot(gs[46:47, 32:62])
    norm2 = mpl.colors.Normalize(vmin=0., vmax=1)
    dbar2 = fig.colorbar(mpl.cm.ScalarMappable(norm=norm2, cmap="jet"),
                         # ticks=[0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85],
                         cax=dx2,
                         orientation="horizontal")
    dbar2.set_label("Fault probability", loc='center', **font1)
    plt.savefig("./compare.png", dpi=300, bbox_inches="tight")



if __name__=="__main__":
    main()
    show_result()

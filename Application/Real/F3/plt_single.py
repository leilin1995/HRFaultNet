"""
__author__: Lei Lin
__project__: plt_single.py
__time__: 2024/9/24 
__email__: leilin1117@outlook.com
"""
# from ImproveResolution.code.model import Generator
from ImproveResolution.code.utils import read_h5, save_h5, normal
# from FaultSegmentation.code.model import Unet
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter
import h5py

mpl.rcParams['font.sans-serif'] = ['Times New Roman']  # font type
mpl.rcParams['axes.unicode_minus'] = False


def read_hdf5(file_path, internal_path="seismic"):
    with h5py.File(file_path, 'r') as file:
        seismic_data = file[internal_path][:]
    return seismic_data


def show_result_single():
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
    fault_Faultseg = read_hdf5("F3_FaultSeg.hdf5", internal_path="predict")
    fault_Swin = read_hdf5("F3_Swin.hdf5",internal_path="predict")
    # fault_raw[fault_raw < 0.3] = 0
    raw_alpha = np.ones_like(fault_raw)
    raw_alpha[fault_raw < 0.55] = 0.5
    raw_alpha[fault_raw <= 0.45] = 0.25
    raw_alpha[fault_raw <= 0.3] = 0.

    faultseg_alpha = np.ones_like(fault_Faultseg)
    faultseg_alpha[fault_Faultseg < 0.55] = 0.5
    faultseg_alpha[fault_Faultseg <= 0.45] = 0.25
    faultseg_alpha[fault_Faultseg <= 0.3] = 0.

    faultswin_alpha = np.ones_like(fault_Swin)
    faultswin_alpha[fault_Swin < 0.55] = 0.5
    faultswin_alpha[fault_Swin <= 0.45] = 0.25
    faultswin_alpha[fault_Swin <= 0.3] = 0.
    # renormal seismic_sr
    seismic_sr = normal(seismic_sr)
    seismic_sr = seismic_sr * (np.max(seismic_raw) - np.min(seismic_raw)) + np.min(seismic_raw)
    print(np.max(seismic_raw), np.min(seismic_raw))
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
    plt.figure(figsize=(8, 6))
    plt.imshow(seismic_raw.T, aspect=1.5, cmap=plt.cm.gray, vmin=-17000, vmax=16000)
    # plt.ylabel("Samples",font)
    # plt.title("Traces", font)
    # plt.text(-12, -14,"(a)",font1)
    plt.savefig("./fig/seismic_raw.png", dpi=300, bbox_inches="tight")

    plt.figure(figsize=(8, 6))
    plt.imshow(seismic_sr.T, aspect=1.5, cmap=plt.cm.gray, vmin=-17000, vmax=16000)
    # plt.ylabel("Samples", font)
    # plt.title("Traces", font)
    # plt.text(-12, -14,"(a)",font1)
    plt.savefig("./fig/seismic_sr.png", dpi=300, bbox_inches="tight")

    # raw fault map
    plt.figure(figsize=(8, 6))
    plt.imshow(seismic_raw.T, aspect=1.5, cmap=plt.cm.gray, vmin=-17000, vmax=16000)
    plt.imshow(fault_raw, aspect=1.5, cmap="jet", vmin=0.15, vmax=0.85, alpha=raw_alpha)
    # plt.ylabel("Samples", font)
    # plt.title("Traces", font)
    # plt.text(-12, -14,"(a)",font1)
    plt.savefig("./fig/fault_raw.png", dpi=300, bbox_inches="tight")

    # FaultSeg3D fault map
    plt.figure(figsize=(8, 6))
    plt.imshow(seismic_raw.T, aspect=1.5, cmap=plt.cm.gray, vmin=-17000, vmax=16000)
    plt.imshow(fault_Faultseg, aspect=1.5, cmap="jet", vmin=0.15, vmax=0.85, alpha=faultseg_alpha)
    # plt.ylabel("Samples", font)
    # plt.title("Traces", font)
    # plt.text(-12, -14,"(a)",font1)
    plt.savefig("./fig/fault_FaultSeg.png", dpi=300, bbox_inches="tight")

    # Swin UNETR fault map
    plt.figure(figsize=(8, 6))
    plt.imshow(seismic_raw.T, aspect=1.5, cmap=plt.cm.gray, vmin=-17000, vmax=16000)
    plt.imshow(fault_Swin, aspect=1.5, cmap="jet", vmin=0.15, vmax=0.85, alpha=faultswin_alpha)
    # plt.ylabel("Samples", font)
    # plt.title("Traces", font)
    # plt.text(-12, -14,"(a)",font1)
    plt.savefig("./fig/fault_SwinUNETR.png", dpi=300, bbox_inches="tight")

    plt.figure(figsize=(8, 6))
    plt.imshow(seismic_sr.T, aspect=1.5, cmap=plt.cm.gray, vmin=-17000, vmax=16000)
    plt.imshow(fault_sr, aspect=1.5, cmap="jet", vmin=0.15, vmax=0.85, alpha=sr_alpha)
    # plt.ylabel("Samples", font)
    # plt.title("Traces", font)
    # plt.text(-12, -14,"(a)",font1)
    plt.savefig("./fig/fault_sr.png", dpi=300, bbox_inches="tight")

    plt.figure(figsize=(8, 6))
    plt.imshow(seismic_raw.T, aspect=1.5, cmap="seismic", vmin=-17000, vmax=16000)
    # plt.ylabel("Samples", font)
    # plt.title("Traces", font)
    # plt.text(-12, -14,"(a)",font1)
    plt.savefig("./fig/seismic_raw_cmap_seis.png", dpi=300, bbox_inches="tight")

    plt.figure(figsize=(8, 6))
    plt.imshow(seismic_sr.T, aspect=1.5, cmap="seismic", vmin=-17000, vmax=16000)
    # plt.ylabel("Samples", font)
    # plt.title("Traces", font)
    # plt.text(-12, -14,"(a)",font1)
    plt.savefig("./fig/seismic_sr_cmap_seis.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    show_result_single()

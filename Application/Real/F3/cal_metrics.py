"""
__author__: Lei Lin
__project__: cal_metrics.py
__time__: 2024/9/27 
__email__: leilin1117@outlook.com
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.ndimage import binary_dilation

def compute_f1_score(pred, label):
    assert pred.shape == label.shape, "Shape of prediction and label must be the same"
    tp = np.logical_and(pred == 1, label == 1).sum()  # True Positives
    fp = np.logical_and(pred == 1, label == 0).sum()  # False Positives
    fn = np.logical_and(pred == 0, label == 1).sum()  # False Negatives

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    if precision + recall == 0:
        return 0

    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score

def compute_miou(pred, label):
    # 确保预测和标签形状相同
    assert pred.shape == label.shape, "Shape of prediction and label must be the same"

    # 初始化 IoU 数组（对于二分类问题，类别数为2）
    iou = []

    for i in range(2):  # 0表示背景，1表示断层
        # 计算交集
        intersection = np.logical_and(pred == i, label == i).sum()
        # 计算并集
        union = np.logical_or(pred == i, label == i).sum()

        if union == 0:
            iou.append(1)  # 如果该类不存在，IoU 设为1
        else:
            iou.append(intersection / union)

    # 计算 MIoU
    miou = np.mean(iou)
    return miou


def compute_foreground_iou(pred, label):
    # 确保预测和标签形状相同
    assert pred.shape == label.shape, "Shape of prediction and label must be the same"

    # 只计算前景（断层，即标签为1）的 IoU
    intersection = np.logical_and(pred == 1, label == 1).sum()
    union = np.logical_or(pred == 1, label == 1).sum()

    if union == 0:
        iou = 1  # 如果前景不存在，IoU 设为1
    else:
        iou = intersection / union

    return iou


def read_hdf5(file_path, internal_path="seismic"):
    with h5py.File(file_path, 'r') as file:
        seismic_data = file[internal_path][:]
    return seismic_data


def main():
    # 读取数据
    label = np.load("F3_resized.npy")
    fault_sr = read_hdf5("./hr_fault.h5", internal_path="/data")
    fault_raw = read_hdf5("./raw_fault.h5", internal_path="/data")
    fault_faultseg = read_hdf5("./F3_FaultSeg.hdf5", internal_path="predict")
    fault_swin = read_hdf5("./F3_Swin.hdf5", internal_path="predict")
    # 转化为01标签，1表示断层
    label[label == 0] = 1
    label[label != 1] = 0
    # 定义结构元素，用于控制膨胀的形状
    structure = np.ones((3, 3))  # 使用2D结构元素

    # 进行膨胀操作，迭代次数为1
    label_dilated = binary_dilation(label, structure=structure, iterations=1)

    # # 可视化膨胀前后的结果
    # fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    #
    # # 原始标签的可视化
    # axes[0].imshow(label, cmap='gray')
    # axes[0].set_title('膨胀前的标签')
    # axes[0].axis('off')
    #
    # # 膨胀后的标签的可视化
    # axes[1].imshow(label_dilated, cmap='gray')
    # axes[1].set_title('膨胀后的标签')
    # axes[1].axis('off')
    #
    # # 显示图像
    # plt.tight_layout()
    # plt.show()

    # 转化为二分类标签
    fault_raw_binary = (fault_raw > 0.5).astype(int)
    fault_sr_binary = (fault_sr > 0.5).astype(int)  # 假设阈值为0.5
    fault_faultseg_binary = (fault_faultseg > 0.5).astype(int)
    fault_swin_binary = (fault_swin > 0.5).astype(int)
    # 计算 MIoU
    miou_faultseg = compute_foreground_iou(fault_faultseg_binary, label_dilated[::2, ::2])
    miou_raw = compute_foreground_iou(fault_raw_binary, label_dilated[::2, ::2])
    miou_swin = compute_foreground_iou(fault_swin_binary, label_dilated[::2, ::2])
    miou_sr = compute_foreground_iou(fault_sr_binary, label_dilated)
    f1_faultseg = compute_f1_score(fault_faultseg_binary, label_dilated[::2, ::2])
    f1_raw = compute_f1_score(fault_raw_binary, label_dilated[::2, ::2])
    f1_swin = compute_f1_score(fault_swin_binary, label_dilated[::2, ::2])
    f1_sr = compute_f1_score(fault_sr_binary, label_dilated)

    print(f"MIoU (FaultSeg): {miou_faultseg}")
    print(f"MIoU (Raw): {miou_raw}")
    print(f"MIoU (Swin): {miou_swin}")
    print(f"MIoU (SR): {miou_sr}")

    print(f"F1 Score (FaultSeg): {f1_faultseg}")
    print(f"F1 Score (Raw): {f1_raw}")
    print(f"F1 Score (Swin): {f1_swin}")
    print(f"F1 Score (SR): {f1_sr}")


if __name__ == "__main__":
    main()

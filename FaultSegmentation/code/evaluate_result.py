"""
__author__ = 'linlei'
__project__:evaluate_result
__time__:2021/9/28 10:31
__email__:"919711601@qq.com"
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import read_h5, ConfusionMatrix
import matplotlib as mpl
import itertools
from sklearn import metrics

mpl.rcParams['font.sans-serif'] = ['Times New Roman']  # font type
mpl.rcParams['axes.unicode_minus'] = False
# from tqdm import tqdm


"""here are six semantic segmentation evaluation index,whice are calculated by confusion matrix
    PA:
        PA is pixel accuracy
        PA = TP + TN / TP +TN + FP + FN

    MPA:
        Calculate the proportion between the correct pixel number of each category and all pixel points of this category, and then average it
        MPA = (TP / (TP + FN) + TN / (TN + FP)) / 2

    PR:
        PR is precision rate
        PR = TP / TP + FP

    RE:
        RE is recall rate
        RE = TP / TP + FN

    Dice:
        Dice coefficient is a measure of set similarity
        Dice = 2TP / FP + 2TP + FN

    IOU:
        Intersection over Union
        IOU = TP / FP + TP + FN

"""


class Evaluator():
    def __init__(self, confusion_matrix, num_classes=2):
        """

        Args:
            confusion_matrix: The confusion matrix obtained from the predicted results and the real results
            num_classes:category,there are two types of fault recognition by default.
        """
        self.num_classes = num_classes
        self.matrix = confusion_matrix

    def Pixel_Accuracy(self):
        PA = np.sum(np.diag(self.matrix)) / np.sum(self.matrix)
        return PA

    def Mean_Pixel_Accuracy(self):
        MPA_array = np.diag(self.matrix) / np.sum(self.matrix, axis=1)
        MPA_array = np.diag(self.matrix) / np.sum(self.matrix, axis=1)
        MPA = 0.2 * MPA_array[0] + 0.8 * MPA_array[1]
        return MPA

    def IOU(self):
        iou = self.matrix[1, 1] / (self.matrix[1, 0] + self.matrix[1, 1] + self.matrix[0, 1])
        return iou

    def Dice(self):
        dice = 2 * self.matrix[1, 1] / (self.matrix[1, 0] + 2 * self.matrix[1, 1] + self.matrix[0, 1])
        return dice

    def Precision_rate(self):
        pr = self.matrix[1, 1] / (self.matrix[1, 1] + self.matrix[0, 1])
        return pr

    def Recall_rate(self):
        re = self.matrix[1, 1] / (self.matrix[1, 1] + self.matrix[1, 0])
        return re

    def F1_score(self):
        pr = self.matrix[1, 1] / (self.matrix[1, 1] + self.matrix[0, 1])
        re = self.matrix[1, 1] / (self.matrix[1, 1] + self.matrix[1, 0])
        F1 = 2 * (pr * re) / (pr + re)
        return F1


def plot_confusion_matrix(confusion_matrix, classes=["No Faults", "Faults"], title="Confusion Matrix",
                          cmap=plt.cm.Blues, normalize=False,save_path="../../result_systhetic_2D/confusion_matrix.png"):
    """

    Args:
        confusion_matrix: The value of the calculated confusion matrix
        classes: Classes of confusion matrix
        title: Figure title
        cmap: Colorbar
        save_path: Save path of .png
    Returns:

    """
    if normalize:
        confusion_matrix = confusion_matrix.astype("float") / confusion_matrix.sum(axis=1)[:, np.newaxis]
    else:
        confusion_matrix = confusion_matrix.astype("int")
    plt.figure()
    plt.imshow(confusion_matrix, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    plt.ylim(len(classes) - 0.5, -0.5)
    thresh = confusion_matrix.max() / 2.
    for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
        plt.text(j, i, format(confusion_matrix[i, j],fmt),
                 horizontalalignment="center",
                 color="white" if confusion_matrix[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.savefig(save_path, dpi=300)


def cal_auc(y_true, y_pred):
    n_bins = 100
    postive_len = sum(y_true)  # M正样本个数
    negative_len = len(y_true) - postive_len  # N负样本个数
    total_case = postive_len * negative_len  # M * N样本对数
    pos_histogram = [0 for _ in range(n_bins)]  # 保存每一个概率值下的正样本个数
    neg_histogram = [0 for _ in range(n_bins)]  # 保存每一个概率值下的负样本个数
    bin_width = 1.0 / n_bins
    for i in range(len(y_true)):
        nth_bin = int(y_pred[i] / bin_width)  # 概率值转化为整数下标
        if y_true[i] == 1:
            pos_histogram[nth_bin] += 1
        else:
            neg_histogram[nth_bin] += 1
    # print(pos_histogram)
    # print(neg_histogram)
    accumulated_neg = 0
    satisfied_pair = 0
    for i in range(n_bins):
        satisfied_pair += (pos_histogram[i] * accumulated_neg + pos_histogram[i] * neg_histogram[i] * 0.5)
        # print(pos_histogram[i], neg_histogram[i], accumulated_neg, satisfied_pair)
        accumulated_neg += neg_histogram[i]
    return satisfied_pair / float(total_case)


def plot_roc(TPR, FPR, AUC, save_path="../../result_systhetic_2D/roc.png"):
    plt.figure(figsize=(8,7))
    plt.rcParams.update({'font.size': 18})
    plt.plot(FPR, TPR, color="red")
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tick_params(labelsize=16)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    # plt.title("Roc curve of AUC: {:.3f}".format(AUC))
    plt.savefig(save_path, dpi=300,bbox_inches="tight")


def get_evaluate(label_folder: str, pred_folder: str, num_classes: int = 2, threshold: float = 0.5,
                 get_roc: bool = True, save_path: str = None):
    """

    Args:
        label_folder: Folders corresponding to real faults
        pred_folder: Folders corresponding to predicted faults
        num_classes: Number of classes
        threshold: Dividing the predicted probability into fault and non-fault thresholds
        get_roc: Whether or not you need to obtain the results of the roc evaluation, which can take a lot of time to calculate
        save_path: Path to save evaluation metrics and images


    Returns:

    """
    label_list = os.listdir(label_folder)
    pred_list = os.listdir(pred_folder)
    assert len(label_list) == len(pred_list), "The number of real images is not equal to the number of predicted images"
    num_list = len(label_list)
    if get_roc:
        print("Begin roc calculated.")
        # roc = {"threshold": [], "TPR": [], "FPR": []}
        all_pred = []
        all_true = []
        for i in range(num_list):
            pred = read_h5(os.path.join(pred_folder, pred_list[i]))
            label = read_h5(os.path.join(label_folder, pred_list[i]))
            label_flatten = list(label.flatten())
            pred_flatten = list(pred.flatten())
            all_pred.extend(pred_flatten)
            all_true.extend(label_flatten)
        fpr, tpr, thresholds = metrics.roc_curve(np.array(all_true), np.array(all_pred))
        auc = metrics.roc_auc_score(np.array(all_true), np.array(all_pred))
        with open(save_path + "/auc.txt", "w") as f:
            f.write("AUC: {:.3f}".format(auc))
            f.close()
        # df = pd.DataFrame(roc)
        # df.to_csv(save_path + "/roc.csv")
        plot_roc(tpr, fpr, auc, save_path=save_path + "/roc.png")
        print("Finished roc calculated.")


    print("Start calculating multiple evaluation metrics.")
    confusion_matrix = ConfusionMatrix(num_classes=num_classes, threshold=threshold)
    for i in range(num_list):
        pred = read_h5(os.path.join(pred_folder, pred_list[i]))
        label = read_h5(os.path.join(label_folder, label_list[i]))
        confusion_matrix.update(pred, label)
    # print("confusion matrix: ", confusion_matrix.matrix)
    # save confusion matrix
    plot_confusion_matrix(confusion_matrix=confusion_matrix.matrix, normalize=False,
                          save_path=save_path + "/confusion_matrix.png")
    plot_confusion_matrix(confusion_matrix=confusion_matrix.matrix, normalize=True,
                          save_path=save_path + "/confusion_normalized_matrix.png")
    # calculate multiple evaluation metrics
    evaluator = Evaluator(confusion_matrix.matrix)
    Pixel_Accuracy = evaluator.Pixel_Accuracy()
    Mean_Pixel_Accuracy = evaluator.Mean_Pixel_Accuracy()
    IOU = evaluator.IOU()
    Dice = evaluator.Dice()
    Precision_rate = evaluator.Precision_rate()
    Recall_rate = evaluator.Recall_rate()
    F1 = evaluator.F1_score()
    print("Finished calculating multiple evaluation metrics.")
    # print(
    #     "Pixel_Accuracy:{:.2f}\nMean_Pixel_Accuracy:{:.2f}\nPrecision_rate{:.2f}\nRecall_rate:{:.2f}\nIOU:{:.2f}\nDice:{:.2f}\nF1_score:{:.2f}".
    #     format(Pixel_Accuracy, Mean_Pixel_Accuracy, Precision_rate, Recall_rate, IOU, Dice, F1))
    # save multiple evaluation metrics
    with open(save_path + "/evaluate.txt", "w") as f:
        f.write(
            "Pixel_Accuracy:{:.3f}\nMean_Pixel_Accuracy:{:.3f}\nPrecision_rate{:.3f}\nRecall_rate:{:.3f}\nIOU:{:.3f}\nDice:{:.3f}\nF1_score:{:.3f}".
            format(Pixel_Accuracy, Mean_Pixel_Accuracy, Precision_rate, Recall_rate, IOU, Dice, F1))
        f.close()


if __name__ == "__main__":
    get_evaluate(label_folder=r"../data/val/fault",
                 pred_folder=r"../result_systhetic_2D/predict", threshold=0.5,
                 save_path=r"../result_systhetic_2D")




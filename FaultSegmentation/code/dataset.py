"""
__author__ = 'linlei'
__project__:dataset
__time__:2021/9/28 11:18
__email__:"919711601@qq.com"
"""
import torch
from torch.utils.data import Dataset
import os
import numpy as np
from utils import read_h5
from PIL import Image
import matplotlib.pyplot  as plt

class MydataSet(Dataset):

    def __init__(self,image_path:str,label_path:str,transform = True):
        """

        Args:
            image_path:seismic images path
            label_path:fault labels path
            transform:use data argumentation or not
        """
        self.image_path = image_path
        self.label_path = label_path
        self.image_list = os.listdir(self.image_path)
        self.label_list = os.listdir(self.label_path)
        self.transform = transform
        assert len(self.image_list) == len(self.label_list),"the number of images is not equal to the number of labels"

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        image = read_h5(os.path.join(self.image_path,self.image_list[item]))    # seismic
        label = read_h5(os.path.join(self.label_path,self.label_list[item]))    # label
        if self.transform:
            rot = np.random.rand()
            flip = np.random.randint(0,3)
            image = self._transform(image,rot = rot,flip = flip).copy()
            label = self._transform(label,rot = rot,flip = flip).copy()
        image = torch.unsqueeze(torch.from_numpy(image).float(),dim = 0)
        label = torch.unsqueeze(torch.from_numpy(label).float(),dim = 0)
        return image,label

    """data argumentation"""
    @staticmethod
    def _transform(data,rot = 0,flip = 0):
        # rotation
        if rot > 0.5:
            data = np.rot90(data,k = 2)
        # flip
        if flip == 0:
            return data
        elif flip == 1:
            data = np.flip(data,axis = 0)
            return data
        elif flip == 2:
            data = np.flip(data,axis = 1)
            return data


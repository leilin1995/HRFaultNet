import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import read_h5,save_h5
from torchvision.transforms import ToTensor,Resize,InterpolationMode
from PIL import Image



class SR_Dataset(Dataset):
    def __init__(self,low_path,high_path,normal = True,argumentation = True,train = True):
        super().__init__()
        self.high_files_path = [os.path.join(high_path,x) for x in os.listdir(high_path)]
        self.low_files_path = [os.path.join(low_path,x) for x in os.listdir(low_path)]
        self.argumentation = argumentation
        self.normal = normal
        self.train = train

    def __getitem__(self,index):
        high_data = read_h5(self.high_files_path[index])
        low_data = read_h5(self.low_files_path[index])
        restore_size = high_data.shape

        if self.argumentation:
            low_data,high_data = self._argumentation(low_data,high_data)

        if self.normal:
            low_data = self._normal(low_data)
            high_data = self._normal(high_data)

        to_tensor = ToTensor()
        if self.train:
            return to_tensor(low_data).type(torch.FloatTensor),to_tensor(high_data).type(torch.FloatTensor)
        else:
            hr_scale = Resize(restore_size, interpolation=InterpolationMode.BICUBIC)
            img_low_data = Image.fromarray(low_data)
            hr_restore_img = np.array(hr_scale(img_low_data))
            return to_tensor(low_data).type(torch.FloatTensor),to_tensor(hr_restore_img).type(torch.FloatTensor),to_tensor(high_data).type(torch.FloatTensor)

    def __len__(self):
        return len(self.high_files_path)

    def _normal(self,data):
        data_max = np.max(data)
        data_min = np.min(data)
        normal = (data - data_min) / (data_max - data_min)
        return normal

    def _argumentation(self,low_data,high_data):
        if random.random() > 0.5:
            high_data = np.flip(high_data, axis=0)
            low_data = np.flip(low_data, axis=0)
        if random.random() > 0.5:
            high_data = np.flip(high_data, axis=1)
            low_data = np.flip(low_data, axis=1)
        if random.random() > 0.5:
            high_data = np.rot90(high_data, k=2)
            low_data = np.rot90(low_data, k=2)
        return low_data,high_data


if __name__ == "__main__":
    train_dataset = SR_Dataset(low_path = "../data/data/SRF_2/train/low",high_path = "../data/data/SRF_2/train/high",train = False)
    print(train_dataset[0][1].size())

"""
__author__ = 'linlei'
__project__:predict
__time__:2021/9/28 10:44
__email__:"919711601@qq.com"
"""
import argparse
import os
import torch
from model import Unet
from utils import read_h5, save_h5
import numpy as np


def main(args):
    # define device
    device = torch.device("cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu")
    # creat save folder
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # build model
    model = Unet(in_ch=args.in_channel, out_ch=args.out_channel).to(device)
    path = args.state_path

    # load state dict
    loader = torch.load(path, map_location=device)
    model.load_state_dict(loader["model_state_dict"])

    # load state dict
    loader = torch.load(path, map_location=device)
    model.load_state_dict(loader["model_state_dict"])

    # batch prediction
    path_list = os.listdir(args.input_path)
    model.eval()
    # image size
    x, y = args.fig_size
    # prediction
    with torch.no_grad():
        for path in path_list:
            # read data and add channel dim
            data = read_h5(os.path.join(args.input_path, path)).reshape(args.in_channel, x, y)
            # to tensor
            data = torch.from_numpy(data).to(device)
            # expand batch dim
            data = torch.unsqueeze(data, dim=0)
            pred = model(data)
            # squeeze batch dim
            pred = torch.squeeze(pred, dim=0).cpu().numpy()
            if args.in_channel == 1:
                pred = np.squeeze(pred, axis=0)
            # save data
            save_h5(data=pred, path=os.path.join(save_path, path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, default="../result_systhetic_2D/predict",
                        help="Save the predicted file")
    parser.add_argument("--state_path", type=str, default="../result_systhetic_2D/best_model.pth",
                        help="The path of the well-trained model")
    parser.add_argument("--input_path", type=str, default="../data/val/seis",
                        help="The path of the data to be predicted")
    parser.add_argument("--device", type=int, default=0, help="Gpu id of used eg. 0,1,2")
    parser.add_argument("--in_channel", type=int, default=1, help="The channel number of input image")
    parser.add_argument("--out_channel", type=int, default=1, help="The channel number of input image")
    parser.add_argument("--fig_size", type=tuple, default=(128, 128), help="Image size for prediction")
    args = parser.parse_args()
    # run
    main(args)

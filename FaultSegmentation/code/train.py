"""
__author__ = 'linlei'
__project__:train
__time__:2021/9/28 10:43
__email__:"919711601@qq.com"
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from dataset import MydataSet
from model import Unet
from utils import FocalLoss
# from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import os
import pandas as pd


def main(args):
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    # save hyperparameter
    argsDict = args.__dict__
    with open(args.save_path + "/hyperparameter.txt", "w") as f:
        f.writelines("-" * 10 + "start" + "-" * 10 + "\n")
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + " :    " + str(value) + "\n")
        f.writelines("-" * 10 + "end" + "-" * 10)
    # device
    device = torch.device("cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu")

    # prepare data
    # dataset
    train_data = MydataSet(image_path=args.train_image_path, label_path=args.train_label_path, transform=args.transform)
    val_data = MydataSet(image_path=args.val_image_path, label_path=args.val_label_path, transform=False)

    # dataloader
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=False)

    # build model,optimizer,loss
    model = Unet(in_ch=args.in_channel, out_ch=args.out_channel).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(args.b1, args.b2))
    criteria = args.loss.to(device)
    epochs = args.epoch
    start_epoch = 1

    best_loss = 10000
    # transfer learning or not
    if args.transfer:
        # load checkpoint
        path_checkpoint = args.checkpoint_path
        load_point = torch.load(path_checkpoint, map_location=device)
        model.load_state_dict(load_point["model_state_dict"])
    loss_log = {"epoch": [], "train_loss": [], "val_loss": []}
    for epoch in range(start_epoch, epochs + 1):

        model.train()
        train_loss = 0.
        # train_loader = tqdm(train_loader)
        train_step = 0.
        # train step
        for step, data in enumerate(train_loader):
            train_step += 1
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            preds = model(images)
            loss = criteria(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss
            # train_loss_step = train_loss / train_step
            # train_loader.set_description(
            #     "train | batch:[{}/{}],epoch:[{}/{}],loss:{:.4f}.".format(step, len(train_loader), epoch, epochs,
            #                                                               train_loss_step))

        train_loss_epoch = train_loss.detach().cpu().numpy() / len(train_loader)

        # val step
        model.eval()
        # val_loader = tqdm(val_loader)
        val_loss = 0.
        val_step = 0.
        with torch.no_grad():
            for step, data in enumerate(val_loader):
                val_step += 1
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)

                preds = model(images)
                loss = criteria(preds, labels)
                val_loss += loss
                # val_loss_step = val_loss / val_step
                # val_loader.set_description(
                #     "val | batch:[{}/{}],epoch:[{}/{}],loss:{:.4f}.".format(step, len(val_loader), epoch, epochs,
                #                                                             val_loss_step))

            val_loss_epoch = val_loss.cpu().numpy() / len(val_loader)
        print("Epoch: {}| train loss: {:.4f},val_loss: {:.4f}".format(epoch, train_loss_epoch, val_loss_epoch))
        if val_loss_epoch < best_loss:
            best_loss = val_loss_epoch
            torch.save({"model_state_dict": model.state_dict()}, args.save_path + "/best_model.pth")

        loss_log["epoch"].append(epoch)
        loss_log["train_loss"].append(train_loss_epoch)
        loss_log["val_loss"].append(val_loss_epoch)

    if args.save_loss:
        frame_loss = pd.DataFrame(
            data=loss_log,
            index=range(1, args.epoch + 1)
        )
        frame_loss.to_csv(args.save_path + "/loss.csv")
        plt_loss(loss_log["epoch"], loss_log["train_loss"], loss_log["val_loss"],
                 save_path=args.save_path + "/loss.png")


def plt_loss(epoch, train_loss, val_loss, save_path):
    plt.figure()
    plt.plot(epoch, train_loss, "r", label="train loss")
    plt.plot(epoch, val_loss, "b", label="val loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(save_path, dpi=100)


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--device", type=str, default=0, help="The number of the GPU used,eg.0,1,2")
    parse.add_argument("--train_image_path", type=str, default="../../systhetic_data/data_2D/train/seis",
                       help="Path of the training set images")
    parse.add_argument("--train_label_path", type=str, default="../../systhetic_data/data_2D/train/fault",
                       help="Path of the training set labels")
    parse.add_argument("--val_image_path", type=str, default="../../systhetic_data/data_2D/val/seis",
                       help="Path of the val set images")
    parse.add_argument("--val_label_path", type=str, default="../../systhetic_data/data_2D/val/fault",
                       help="Path of the val set labels")
    parse.add_argument("--transform", type=bool, default=True, help="Whether to use data augmentation")
    parse.add_argument("--batch_size", type=int, default=4, help="Batch size used for training")
    parse.add_argument("--in_channel", type=int, default=1, help="Number of slices of input data")
    parse.add_argument("--out_channel", type=int, default=1, help="Number of slices of output data")
    parse.add_argument("--learning_rate", type=float, default=1e-5,
                       help="The value of the learning rate used for training")
    parse.add_argument("--loss", default=FocalLoss(), help="loss function")
    parse.add_argument("--transfer", type=bool, default=False, help="use pre-training or not")
    parse.add_argument("--checkpoint_path", type=str, help="Path to pre-trained model")
    parse.add_argument("--epoch", type=int, default=200, help="Training epochs")
    parse.add_argument("--b1", type=float, default=0.9, help="Adam optimizer hyparameter")
    parse.add_argument("--b2", type=float, default=0.999, help="Adam optimizer hyparameter")
    parse.add_argument("--save_loss", type=bool, default=True, help="Save loss log or not")
    parse.add_argument("--save_path", type=str, default="../../result_systhetic_2D",
                       help="Path to save the model and loss")
    args = parse.parse_args()
    main(args)

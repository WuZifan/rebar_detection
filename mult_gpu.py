#!/usr/bin/python3
# coding: utf-8
import torch
from torchvision import datasets, transforms
import torchvision
from tqdm import tqdm

device_ids = [1, 2, 3]
BATCH_SIZE = 8



class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=2, kernel_size=2),
        )
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(14 * 14 * 128, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(1024, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 14 * 14 * 128)
        x = self.dense(x)
        return x


model = Model()
model = torch.nn.DataParallel(model, device_ids=device_ids)  # 声明所有可用设备
model = model.cuda(device=device_ids[0])  # 模型放在主设备

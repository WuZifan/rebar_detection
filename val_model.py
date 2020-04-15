from __future__ import division

from deep_model.models import *
from utils.utils import *
from utils.datasets import *
from utils.logger import *

from test import evaluate

import os
import sys
import time
import datetime
import argparse

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

from terminaltables import AsciiTable


class Rebar_Yolov3:

    _default = {
        'class_path':"my_data/rebar.names",
        'model_def':"my_data/yolov3_rebar.cfg",
        'weights_path':"weights/yolov3_ckpt_99.pth",
        'conf_thres':0.8,
        'nms_thres':0.4,
        'batch_size':1,
        'n_cpu':0,
        'img_size':416,
    }

    def __init__(self):
        self.__dict__.update(self._default)
        self.classes = load_classes(self.class_path)  # Extracts class labels from file
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model =self.get_model()

    def get_model(self):
        '''
        拿到模型
        :return:
        '''
        # Set up model
        model = Darknet(self.model_def, img_size=self.img_size).to(self.device)
        return model

    def load_weights(self):
        '''
        加载权重
        :return:
        '''
        if self.weights_path.endswith(".weights"):
            # Load darknet weights
            self.model.load_darknet_weights(self.weights_path)
        else:
            # Load checkpoint weights
            self.model.load_state_dict(torch.load(self.weights_path, map_location=self.device))

    def detect(self,image,display=True):
        '''
        检测
        :param image:
        :param display:
        :return:
        '''
        self.load_weights() #加载权重
        self.model.eval() # 不加这句话BN层和Dropout层不固定
        prev_time = time.time()
        if isinstance(image,np.ndarray):
            image = Image.fromarray(image)

        if isinstance(image,str):
            image = Image.open(image)

        org_w, org_h = image.size

        # 预处理
        img = self.preprocess(image)
        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        input_imgs = Variable(img.type(Tensor))

        '''
            nms最后输出的是[res1,res2...],
            res1: [x1,y1,x2,y2,conf,cls_conf,cls_pred]
        '''
        with torch.no_grad():
            # 原始的输出坐标是(center x, center y, width, height)
            detections = self.model(input_imgs)
            print('before nms',detections.shape)
            print('nms param',self.conf_thres,self.nms_thres)
            # nms中会转换成(x1, y1, x2, y2)
            detections = non_max_suppression(detections, self.conf_thres, self.nms_thres)[0]
            print('after nms',detections.shape)


        print('det shape: ',detections.shape)

        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print("\t+ Inference Time: %s" % (inference_time))

        # 将结果从416*416的输入图像映射回原图上
        if detections is not None:
            detections = rescale_boxes(detections, self.img_size, (org_h, org_w))
            unique_labels = detections[:, -1].cpu().unique()

        if display:
            self.display(detections, image)

        return detections

    def display(self,detections,image):
        '''
        可视化
        :param detections:
        :param image:
        :return:
        '''
        # 可视化
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(image)
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            # print("\t+ Label: %s, Conf: %.5f" % (self.classes[int(cls_pred)], cls_conf.item()))

            box_w = x2 - x1
            box_h = y2 - y1

            color = [random.random() for i in range(3)]
            color.append(1)
            color = tuple(color)
            # Create a Rectangle patch
            bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
            # Add the bbox to the plot
            ax.add_patch(bbox)
            # Add label
            # plt.text(
            #     x1,
            #     y1,
            #     s=self.classes[int(cls_pred)],
            #     color="white",
            #     verticalalignment="top",
            #     bbox={"color": color, "pad": 0},
            # )

        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        plt.show()

        # 考虑是否保存
        # plt.savefig(f"./dog.png", bbox_inches="tight", pad_inches=0.0)
        plt.close()

    def preprocess(self,image):
        '''
        预处理
        :param image:
        :return:
        '''
        img = transforms.ToTensor()(image)
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)
        img.unsqueeze_(0)
        return img


if __name__ == '__main__':

    rebar_yolo = Rebar_Yolov3()
    rebar_yolo.detect('./data/samples/8ADCAE58.jpg',display=True)




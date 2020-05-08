from __future__ import division

from deep_model.models import *
from utils.logger import *
from utils.mylogger import MyLogger
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from test import evaluate

from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'

#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=400, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    # parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    # parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")

    parser.add_argument("--model_def", type=str, default="my_data/yolov3_rebar_asff.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="my_data/rebar.data", help="path to data config file")

    #parser.add_argument("--pretrained_weights",default = 'weights/v1_yolov3_ckpt_99.pth', type=str, help="if specified starts from checkpoint model")
    #parser.add_argument("--pretrained_weights",default='weights/darknet53.conv.74',type=str, help="if specified starts from checkpoint model")
    #parser.add_argument("--pretrained_weights",default='checkpoints/yolov3_ckpt_146_0.9405248761177063.pth',type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--pretrained_weights",default=None,type=str, help="if specified starts from checkpoint model")

    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=2, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    parser.add_argument("--finetune", default=False, help="finetune or not ")
    opt = parser.parse_args()
    print(opt)

    logger = Logger("logs")
    my_logger = MyLogger('./logs/just_logs/train.log',logger_name='myyolov3_trainer')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Get data configuration
    data_config = parse_data_config(opt.data_config)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])
    print(class_names)

    #########################
    # 这部分需要注意，是finetune，是继续训练还是从头训练。
    # Initiate model

    model = Darknet(opt.model_def)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
         # 就这一行
        model = nn.DataParallel(model).cuda()
        import time
        time.sleep(10)



    model.apply(weights_init_normal)

    # If specified we start from checkpoint
    if opt.pretrained_weights:
        print('pre train')
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights),False)
        else:
            model.load_darknet_weights(opt.pretrained_weights)
    
 
    if opt.finetune:
        pretrained_parameters = []
        classifier_parameters = []
        for name,layer in model.named_parameters():
            print(name,layer.shape)
            layer_num = name.split('.')[2]
            if int(layer_num)<74:
                pretrained_parameters.append(layer)
            else:
                classifier_parameters.append(layer)

        print(len(pretrained_parameters),len(classifier_parameters))
        print(type(model.parameters()))

    # optimizer = torch.optim.Adam(model.parameters())
        optimizer = torch.optim.Adam([
            {'params':pretrained_parameters,'lr':1e-4},
            {'params':classifier_parameters,'lr':1e-3}
        ])
    else:
        optimizer = torch.optim.Adam(model.parameters())

    ##########################
    # time.sleep(1000)

    # Get dataloader
    dataset = ListDataset(train_path, augment=True, multiscale=opt.multiscale_training)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    for epoch in range(opt.epochs):
        model.train()
        start_time = time.time()
        for batch_i, (_, imgs, targets) in tqdm(enumerate(dataloader)):
            batches_done = len(dataloader) * epoch + batch_i
            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)

            print('imgs,targets: ',imgs.is_cuda,targets.is_cuda)

            loss, outputs = model(imgs, targets)
            print(type(loss))
            print(type(outputs))
            loss.backward()

           

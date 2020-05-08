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
    parser.add_argument("--pretrained_weights",default='checkpoints/yolov3_ckpt_146_0.9405248761177063.pth',type=str, help="if specified starts from checkpoint model")

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
    model = Darknet(opt.model_def).to(device)
    model.apply(weights_init_normal)

    

    # If specified we start from checkpoint
    if opt.pretrained_weights:
        print('pre train')
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights),False)
        else:
            model.load_darknet_weights(opt.pretrained_weights)
    
    #device_ids = [0, 1, 2, 3]
    #model = torch.nn.DataParallel(model, device_ids=device_ids) # 声明所有可用设备
    #model = model.cuda(device=device_ids[0])  # 模型放在主设备
    # PS的方式加载模型
    model = nn.DataParallel(model)


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

            print(imgs.shape)

            loss, outputs = model(imgs, targets)
            print(type(loss))
            print(type(outputs))
            loss.mean().backward()

            if batches_done % opt.gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------

            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i, len(dataloader))

            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.module.yolo_layers))]]]

            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.module.yolo_layers]
                metric_table += [[metric, *row_metrics]]

                # Tensorboard logging
                tensorboard_log = []
                for j, yolo in enumerate(model.module.yolo_layers):
                    for name, metric in yolo.metrics.items():
                        if name != "grid_size":
                            tensorboard_log += [(f"{name}_{j+1}", metric)]
                tensorboard_log += [("loss", loss.mean().item())]
                logger.list_of_scalars_summary(tensorboard_log, batches_done)

            log_str += AsciiTable(metric_table).table
            log_str += f"\nTotal loss {loss.mean().item()}"

            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"\n---- ETA {time_left}"

            my_logger.info(log_str)

            model.module.seen += imgs.size(0)
        
        AP=None
        if epoch % opt.evaluation_interval == 0:
            
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            precision, recall, AP, f1, ap_class = evaluate(
                model,
                path=valid_path,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=opt.img_size,
                batch_size=4,
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]
            logger.list_of_scalars_summary(evaluation_metrics, epoch)

            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            my_logger.info('{}'.format(AsciiTable(ap_table).table))
            my_logger.info('---- mAP {}'.format(AP.mean()))
            print(f"---- mAP {AP.mean()}")

        if epoch % opt.checkpoint_interval == 0 :
            if AP is not None:
                torch.save(model.state_dict(), "checkpoints/yolov3_ckpt_{}_{}.pth".format(epoch,AP.mean()))
            else:
                torch.save(model.state_dict(), "checkpoints/yolov3_ckpt_{}_{}.pth".format(epoch,loss))

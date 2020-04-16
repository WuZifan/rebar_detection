from deep_model.models import *
import traceback
import torch


class MyNN(torch.nn.Module):

    def __init__(self,yolo):
        super(MyNN, self).__init__()
        self.temp_sequential = torch.nn.Sequential()
        for i, md in enumerate(yolo.module_list):
            self.temp_sequential.add_module('Sequential_{}'.format(i), md)


    def forward(self, x):
        return self.temp_sequential(x)


if __name__ == '__main__':
    model_def='my_data/yolov3_rebar.cfg'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model = Darknet(model_def).to(device)
    x = Variable(torch.randn(1,3,416,416))

    torch.onnx.export(model, x, "yolov3.onnx", verbose=True, input_names='raw_image',
                      output_names='x')




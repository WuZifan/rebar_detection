## pytorch模型可视化方案：

1、仿照以下代码，将pytorch模型以onnx模型输出

```python
    
    model_def='my_data/yolov3_rebar.cfg'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model = Darknet(model_def).to(device)
    x = Variable(torch.randn(1,3,416,416))

    torch.onnx.export(model, x, "yolov3.onnx", verbose=True, input_names='raw_image',
                      output_names='x')

```

2、利用Netron软件，进行加载可视化


## ASFF

ASFF的原理可以认为是对FPN的不同level输出，进行了一次加权融合后在输出。

那么结合当前darknet的模型创建方式，进行修改如下：

* 1）配置文件修改（yolov3_rebar_asff.cfg)


    1、将原来[yolo]层的位置全部提取出来，并放在配置文件最后
    2、在原来[yolo]层的位置设置[save]层，以维持原有的shortcut，将原来yolo层的输入保存起来。
    3、在[yolo]层中加入level属性，表示这个[yolo]层是第几个level
    

* 2）models.py


    1、在create_module方法中，仿照'route'层，给save层加入一个'emptylayer'
    2、在Darknet中：
        2.0 注意，所有的nn.Module，都需要被放在nn.moduleList或者nn.Sequential中，不然在转onnx时，会报：
            RuntimeError: Cannot insert a Tensor that requires grad as a constant. 
        2.1 在初始化方法中,创建一个名为asff_module_lsit的moduleList对象，按顺序存储三个ASFF层。
        2.2 在forward中，创建一个asff_inputs，用来保存asff模块的输入
        2.3 在yolo该被执行的时候，从asff_inputs拿到所有的3个输入，从asff_module_lsit中拿到对应层的asffmodule，执行。
        
   


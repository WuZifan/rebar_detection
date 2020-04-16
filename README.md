##pytorch模型可视化方案：

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
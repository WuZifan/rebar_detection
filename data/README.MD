

生成coco形态的数据集步骤。

0、将图片放在./images下，将train_labels.csv放在./目录下
1、创建./labels文件夹。
2、运行format_convert.py转化格式。
3、利用myAugmentation.py进行数据增强
4、利用train_val_divid.py生成训练集和验证集


5、在datasets.py里面，有写怎么根据图片路径获取label路劲的方法，所以这一点不用我们操心。

本文件夹下的数据格式

```angular2html

├── images
│   └── 0B4F3CC3.jpg
├── labels
│   └── 0B4F3CC3.txt
├── train_data.txt
├── val_data.txt
├── *.py

```
'''
    将数据集分成训练和验证两部分
'''


import os
import random


def train_val(img_path,rate):

    img_list = [ os.path.join(img_path,ip) for ip in os.listdir(img_path)]

    random.shuffle(img_list)

    cut_point = int(len(img_list)*rate)

    train_data = [os.path.abspath(a) for a in img_list[:cut_point]]
    val_data = [os.path.abspath(a) for a in img_list[cut_point:]]
    with open('train_dataset.txt','w') as f:
        f.write('\n'.join(train_data))


    with open('val_dataset.txt','w') as f:
        f.write('\n'.join(val_data))


if __name__ == '__main__':
    img_path = './images'
    train_val(img_path,0.9)





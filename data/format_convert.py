'''
    将rebar对应的label格式转换成coco数据格式
    rebar格式：图片名,xmin,ymin,xmax,ymax
    coco格式：class,cx,cy,dw,dy （文件名为图片名称，x,y,w,h均关于图像做了归一化）

    分几步：
    1、train_dataset.txt : 将所有训练集的图片路径写入一个文件。
    2、train_dataset : 将所有训练集的图片放入该文件夹。
    3、train_labels_files : 将所有训练集的label文件放入该文件夹。
'''
import csv
import cv2
import os
import numpy as np


if __name__ == '__main__':

    # 所有图像的地址
    img_root = './images/'
    # label文件
    label_path = './train_labels.csv'
    # 转换后label文件输出地址
    label_file_dir = './labels/'
    # 转换后训练集图像的地址
    img_file_txt = './images.txt'

    '''
        1、将所有的图片路径写入一个文件
    '''
    if not os.path.exists(img_file_txt):
        with open(img_file_txt,'w') as f:
            content = '\n'.join([os.path.join(img_root,temp_img) for temp_img in os.listdir(img_root)])
            f.write(content+'\n')


    '''
        创建放label文件的文件夹
    '''

    if not os.path.exists(label_file_dir):
        os.makedirs(label_file_dir)

    '''
        label转换&保存
        原始格式 x1,y1,x2,y2
        现在格式 label_id,cx1/w,cy1/h,bw/w,by/h
    '''
    label_data = {}
    with open(label_path)as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            try:
                img_name = row[0]
                img_box = row[-1]
                box_label = np.asarray(img_box.split(),dtype=np.float16)
                box_label = np.insert(box_label,0,values=0)

                if img_name not in label_data.keys():
                    label_data[img_name] = [box_label]
                else:
                    label_data[img_name].append(box_label)
            except Exception as e:
                print(e)
                continue

    for key,item in label_data.items():
        item=np.array(item)
        center_x = (item[:,1]+item[:,3])/2
        center_y = (item[:,2]+item[:,4])/2
        box_w = (item[:,3]-item[:,1])
        box_h = (item[:,4]-item[:,2])

        img = cv2.imread(os.path.join(img_root, key))
        h, w, c = img.shape

        center_x = center_x/w
        center_y = center_y/h
        box_w = box_w/w
        box_h = box_h/h

        item[:,1]=center_x
        item[:,2]=center_y
        item[:,3]=box_w
        item[:,4]=box_h

        print(key,item.shape)
        temp_labelfile_name = key.replace('jpg','txt')
        with open(os.path.join(label_file_dir,temp_labelfile_name),'w') as f:
            for li in item:
                li = str(li)
                li=li.replace('[','').replace(']','')
                li = ' '.join(li.split())
                f.writelines(li+'\n')


    print(img.shape)
from utils.augmentations import MyAugmentation
import os
from PIL import Image
import numpy as np
import random

if __name__ == '__main__':

    img_dir = './images'
    img_paths = [ os.path.join(img_dir,img) for img in os.listdir(img_dir)]

    ma = MyAugmentation()

    for _ in range(250):

        img_path  = random.sample(img_paths,1)[0]
        print(img_path)
        # for img_path in img_paths:
        img = Image.open(img_path)
        label = np.loadtxt(img_path.replace('images','labels').replace('.jpg','.txt')).reshape(-1, 5)

        '''
            0、随机旋转度数
            1、随机上下翻转，
            2、随机左右翻转
            3、随机裁剪
        '''
        # 随机旋转
        new_img,new_label = ma.randomRotate(img,label)
        # 上下翻转
        if random.random()>0.5:
            new_img,new_label = ma.verticalFlip(new_img,new_label)
        # 左右翻转
        if random.random()>0.5:
            new_img,new_label = ma.horizontalFlip(new_img,new_label)
        # 随机裁剪
        floor,ceil = 0.4*min(new_img.size),1*min(new_img.size)
        size  =random.randint(floor,ceil)
        print(size)
        new_img,new_label = ma.randomCrop(img,label,size)

        ma.save(new_img,new_label,'./images','./labels')





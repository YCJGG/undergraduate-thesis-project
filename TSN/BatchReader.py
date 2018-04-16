import numpy as np
import torch.utils.data as data
import torch
import os
from numpy.random import randint
from PIL import Image
class DatasetProcessing(data.Dataset):
    def __init__(self, data_path, transform = None):
        #self.feature_path = data_path
        self.segmet = 3
        self.transform = transform
        ffp = open(data_path,'r')
        fp = list(ffp)
        self.video_path = [x.strip().split(' ')[0] for x in fp]
        self.labels = [int(x.strip().split(' ')[1]) for x in fp]
        ffp.close()
        
        
    def __getitem__(self, index):
        path = self.video_path[index]
        path = path.replace('~',r'/home/zhangjingyi')
        dir = os.listdir(path)
        length = len(dir)
        duration = length // 3
        ind_1 =  randint(duration)
        ind_2 = randint(duration) + duration
        ind_3 = randint(length - 2*duration) + 2*duration
        ind_4 = randint(length)
        im1_pth = path + '/' + dir[ind_1]
        im2_pth = path + '/' + dir[ind_2]
        im3_pth = path + '/' + dir[ind_3]
        im4_pth = path + '/' + dir[ind_4]
        img_1 = Image.open(im1_pth)
        img_2 = Image.open(im2_pth)
        img_3 = Image.open(im3_pth)
        img_4 = Image.open(im4_pth)

        img_1 = img_1.convert('RGB')
        img_2 = img_2.convert('RGB')
        img_3 = img_3.convert('RGB')
        img_4 = img_4.convert('RGB')
        if self.transform is not None:
            img_1 = self.transform(img_1)
            img_2 = self.transform(img_2)
            img_3 = self.transform(img_3)
            img_4 = self.transform(img_4)
        label = torch.LongTensor([self.labels[index]]) 
        return img_1, img_2, img_3,img_4, label, index       
        #print(length,ind_1,ind_2,ind_3)
          
    def __len__(self):
        return len(self.video_path)
# a = DatasetProcessing('test.list')
# for i in a:
#     pass

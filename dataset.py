from __future__ import print_function
import torch
# from torch._C import float32
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import scipy.io as sio
from tqdm import tqdm
from tqdm import trange
import random
import pandas as pd
from PIL import Image
import transforms
from cutout import Cutout
import aug_lib

class EmotionDataset(Dataset):
    def __init__(self, root, transform = None, train=True):
        self.train = train
        self.transform = transform
        self.rawdata = pd.read_csv(root)
        self.rawdata = np.array(self.rawdata)
        imgs = self.rawdata[:,1]
        target = self.rawdata[:,0].reshape(-1,1)
        imgs = np.array([list(map(int, img.split(' '))) for img in imgs]).reshape(-1,48,48)
        imgs = imgs/255
        if self.train:
            self.data = imgs[np.where(self.rawdata[:,-1]=='Training')]
            self.target = target[np.where(self.rawdata[:,-1]=='Training')[0]].reshape(-1,1)
        else:
            self.data = imgs[np.where(self.rawdata[:,-1]=='Test')]
            self.target = target[np.where(self.rawdata[:,-1]=='Test')[0]].reshape(-1,1)
    
    def __getitem__(self, index):
        img, target = self.data[index], self.target[index][0]
        # img = img.reshape(1,48,48)
        # img_copy = img.copy()
        # img = np.concatenate((img,img_copy),axis=0)
        # img = np.concatenate((img,img_copy),axis=0)
        # img = img.transpose((1,2,0))
        # img = img.astype(np.uint32)
        # print(img.shape)
        if self.transform is not None:
            im = Image.fromarray(img)
            im = self.transform(im)
            
        return im, target
    
    def __len__(self):
        return len(self.data)
    
# dataset = EmotionDataset('/home/yu-jw19/venom/project2/data/emotion.csv')
# print(dataset.data.shape)
# print(dataset.target.shape)
if __name__=='__main__':
    transform_train = transforms.Compose([
                    transforms.RandomCrop(48, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ])
    augmentpolicy = aug_lib.RandAugment(n = 2, m = 10)
    transform_train.transforms.insert(0, augmentpolicy)
    transform_train.transforms.append(aug_lib.cutoutdefault(4))
    train_loader = torch.utils.data.DataLoader(
        EmotionDataset('/home/yu-jw19/venom/project2/data/emotion.csv',transform=transform_train, train=True),batch_size=32, shuffle=True)
    for x, (i, data) in enumerate(train_loader):
    # i表示第几个batch， data表示该batch对应的数据，包含data和对应的labels
        # print("第 {} 个Batch \n{}".format(i, data))
        count = 0
        for j in i:
            count += 1
            im = transforms.ToPILImage()(j)
            im = im.save("./saved_pic/{}.jpg".format(count))
        dfdfd
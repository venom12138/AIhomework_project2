from __future__ import print_function
import torch
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

class EmotionDataset(Dataset):
    def __init__(self, root, transform = None, train=True):
        self.train = train
        self.transform = transform
        self.rawdata = pd.read_csv(root)
        self.rawdata = np.array(self.rawdata)
        imgs = self.rawdata[:,1]
        target = self.rawdata[:,0].reshape(-1,1)
        imgs = np.array([list(map(int, img.split(' '))) for img in imgs]).reshape(-1,48,48)
        if self.train:
            self.data = imgs[np.where(self.rawdata[:,-1]=='Training')]
            self.target = target[np.where(self.rawdata[:,-1]=='Training')[0]]
        else:
            self.data = imgs[np.where(self.rawdata[:,-1]=='Test')]
            self.target = target[np.where(self.rawdata[:,-1]=='Test')[0]]
    
    def __getitem__(self, index):
        
        img, target = self.data[index], self.target[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target
    
    def __len__(self):
        return len(self.data)
    
# dataset = EmotionDataset('/home/yu-jw19/venom/project2/data/emotion.csv')
# print(dataset.data.shape)
# print(dataset.target.shape)
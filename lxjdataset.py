from __future__ import print_function
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import os
import numpy as np
import scipy.io as sio
from tqdm import tqdm
from tqdm import trange
import pickle
import random



def has_file_allowed_extension(filename, extensions):  #检查输入是否是规定的扩展名
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))] #获取root目录下所有的文件夹名称

    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))} #生成类别名称与类别id的对应Dictionary
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)#将~和~user转化为用户目录，对参数中出现~进行处理
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)): #os.work包含三个部分，root代表该目录路径 _代表该路径下的文件夹名称集合，fnames代表该路径下的文件名称集合
            for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)    #生成（训练样本图像目录，训练样本所属类别）的元组

    return images   #返回上述元组的列

class new_Dataset(datasets.ImageFolder):
    def __init__(self, dir, is_trn, transform = None, *arg, **kw):
        super(new_Dataset, self).__init__(dir, transform)
        classes, class_to_idx = find_classes(dir)
        imgs = make_dataset(dir, class_to_idx)

        self.imgs=imgs
        self.is_trn = is_trn
        self.list_nn, self.new_dist_nn = self.load_nn(dir)

    @staticmethod
    def load_nn(dir):
        if os.path.isfile(os.path.join(dir, 'list_nn.pkl')):
            with open(os.path.join(dir,'list_nn.pkl'),'rb') as f:
                list_nn = pickle.load(f)
            list_nn['trn'] = list_nn['trn']     # python base 0
            list_nn['val'] = list_nn['val']     # python base 0
            with open(os.path.join(dir,'dist_nn.pkl'),'rb') as f:
                dist_nn = pickle.load(f)
            with open(os.path.join(dir,'new_dist_nn.pkl'),'rb') as f:
                new_dist_nn = pickle.load(f)
        # new_dist_nn = np.zeros((list_nn['trn'][0].size+1,list_nn['trn'][0].size+1))
        # for i in range(list_nn['trn'][0].size+1):
        #     zipped = list(zip(list_nn['trn'][i],dist_nn['trn'][i]))
        #     zipped.sort(key=lambda x:x[0])
        #     temp = list(zip(*zipped))
        #     temp = list(temp[1])
        #     temp.insert(i,0)
        #     new_dist_nn[i,:]=temp  
        #     print(i)
        
        # 使用pickle模块存储对象
        # output = open(os.path.join(dir,'new_dist_nn.pkl'), 'wb')
        # pickle.dump(new_dist_nn, output,-1)
        # output.close()
        return list_nn, new_dist_nn

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (img, target) where target is class_index of the input image.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return img, index


    def __len__(self):
        return len(self.list_nn['trn'][0])
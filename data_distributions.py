from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
# import seaborn as sns
from data_utils import load_tiny_imagenet
from utils import *

import os
import json


class MNIST(object):
    def __init__(self):
        data_dir = "./data/"
        
        fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        trX = loaded[16:].reshape((60000,28,28,1)).astype(np.float)

        fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        trY = loaded[8:].reshape((60000)).astype(np.float)

        fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        teX = loaded[16:].reshape((10000,28,28,1)).astype(np.float)

        fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        teY = loaded[8:].reshape((10000)).astype(np.float)

        trY = np.asarray(trY)
        teY = np.asarray(teY)

        y = np.concatenate((trY, teY), axis=0)
        X = np.concatenate((trX, teX), axis=0)
        
        np.random.seed(12)
        np.random.shuffle(X)
        
        np.random.seed(12)
        np.random.shuffle(y)

        self.X = X/255.0
        self.y = y

        self.image_height = 28
        self.image_width = 28
        self.image_channels = 1
        self.N = self.X.shape[0]


    def sample(self, N, out_y=False):
        indices = np.random.choice(range(self.X.shape[0]), N)
        # return self.X[indices].reshape([N, 28, 28, 1])
        if out_y:
            return self.X[indices], self.y[indices]
            
        return self.X[indices]



class ImageNet(object):
    def __init__(self):
        data_dir = "./data/"
        
        data = load_tiny_imagenet('data/tiny-imagenet-100-A', subtract_mean=True, synset_count=20)

        data['X_train'] = np.moveaxis(data['X_train'], 1, 3)
        np.random.shuffle(data['X_train'])

        self.X = data['X_train']/255.0

        self.image_height = 28
        self.image_width = 28
        self.image_channels = 3
        self.N = self.X.shape[0]


    def sample(self, N, out_y=False):
        indices = np.random.choice(range(self.X.shape[0]), N)
        if out_y:
            return self.X[indices], self.y[indices]
            
        return self.X[indices]



class CelebA(object):
    def __init__(self):
        data_dir = "./data/"
        
        input_fname_pattern='*.jpg'

        data = glob(os.path.join("./data", "celebA", input_fname_pattern))
        # data = imread(data[0]);

        self.N = len(data)

        self.data = data

        self.image_height = 64
        self.image_width = 64
        self.image_channels = 3
        

    def sample(self, N, out_y=False):
        # sample_files = self.data[0:N]
        sample_files = np.random.choice(self.data, N)
        sample = [
          get_image(sample_file,
                    input_height=108,
                    input_width=108,
                    resize_height=64,
                    resize_width=64,
                    crop=True,
                    grayscale=False) for sample_file in sample_files]
        
        sample_inputs = np.array(sample).astype(np.float32)
        sample_inputs = (sample_inputs + 1)/2

        return sample_inputs

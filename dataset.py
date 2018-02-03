#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May  6 19:42:37 2017

@author: ldy
"""

from concurrent.futures import ThreadPoolExecutor
import time

import torch
import torch.utils.data as data
import torchvision
from os import listdir
from os.path import join
from collections import deque
from PIL import Image, ImageFilter
import numpy as np

from torchvision.transforms import ToTensor, Compose, RandomCrop, RandomResizedCrop, Resize, RandomRotation, RandomHorizontalFlip, CenterCrop

def save_image(name, out):
    torchvision.utils.save_image((out * 2) - 1, name)

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print('*** timer:%r  %2.2f ms' % \
                (method.__name__, (te - ts) * 1000))
        return result
    return timed

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])

class GaussianNoise(object):
    def __init__(self, stddev):
        self.stddev = stddev
    
    def __call__(self, im):
        return im + self.stddev * torch.randn(*im.shape)
    
class Random90Rotation(object):
    def __call__(self, im):
        choice = np.random.randint(4)
        if choice == 1:
            im = im.transpose(Image.ROTATE_90)
        elif choice == 2:
            im = im.transpose(Image.ROTATE_180)
        elif choice == 3:
            im = im.transpose(Image.ROTATE_270)
        return im

Loader = Compose((
    RandomCrop(300),
    RandomRotation(10, resample=Image.BILINEAR),
    CenterCrop(256),
    RandomHorizontalFlip(),
    Random90Rotation(),
    RandomResizedCrop(128, scale=(0.5, 1.0), ratio=(1., 1.)),
))

def Downscaler(n):
    return Compose((
        lambda im: im.filter(ImageFilter.GaussianBlur(n)),
        Resize((64, 64), interpolation=Image.BICUBIC),
        ToTensor(),
        GaussianNoise(0.04),
    ))

def load_img(filepath):
    img = Image.open(filepath).convert('YCbCr')
    y, _, _ = img.split()
    return y

class DatasetFromFolder(data.Dataset):
    @timeit
    def __init__(self, image_dir):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        self.images = [load_img(fn) for fn in self.image_filenames]
        #self.executor = ThreadPoolExecutor()
        #self.images = list(self.executor.map(load_img, self.image_filenames))
        #self.queue = deque(maxlen=256)
        #self.requeue()

    def requeue(self):
        for _ in range(self.queue.maxlen - len(self.queue)):
            self.queue.append(self.executor.submit(self.make_pair, self.random_image()))

    def random_image(self):
        return np.random.choice(len(self.images))

    def make_pair(self, image_idx):
        orig = Loader(self.images[image_idx])
        hr = ToTensor()(orig)
        lr = Downscaler(np.random.choice([1, 3]))(orig)

        #if index == 0 and np.random.rand() < 0.01:
        #    orig.save('data_orig.png')
        #    save_image('data_hr.png', hr)
        #    save_image('data_lr.png', lr)

        return lr, hr
    
    def __getitem__(self, index):
        return self.make_pair(self.random_image())
        result = self.queue.popleft()
        self.requeue()
        if not result.done():
            print('delay!!')
        return result.result()

    def __len__(self):
        return 64

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
from os import listdir
from os.path import join
from PIL import Image, ImageFilter
import numpy as np

from torchvision.transforms import ToTensor, Compose, RandomCrop, RandomResizedCrop, Resize, RandomRotation, RandomHorizontalFlip, RandomVerticalFlip, CenterCrop

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

Loader = Compose((
    RandomCrop(256),
    RandomRotation(10, resample=Image.BILINEAR),
    CenterCrop(192),
    RandomHorizontalFlip(),
    RandomVerticalFlip(),
    RandomResizedCrop(128, scale=(0.7, 1.0), ratio=(1., 1.)),
))

def Downscaler(n):
    return Compose((
        lambda im: im.filter(ImageFilter.GaussianBlur(n)),
        Resize((64, 64), interpolation=Image.BICUBIC),
        ToTensor(),
        GaussianNoise(10.),
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
        with ThreadPoolExecutor(max_workers=4) as executor:
            self.images = list(executor.map(load_img, self.image_filenames))

    # @timeit
    def __getitem__(self, index):
        orig = Loader(self.images[np.random.choice(len(self.images))])
        # if index == 0: orig.save('orig.png')
        hr = ToTensor()(orig)
        lr = Downscaler(np.random.choice([1, 3, 5]))(orig)

        return lr, hr

    def __len__(self):
        return 64
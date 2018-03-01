#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May  6 19:42:37 2017

@author: ldy
"""

from concurrent.futures import ThreadPoolExecutor as Executor
import time
import threading

from os import listdir
from os.path import join
from queue import Queue
from PIL import Image, ImageFilter, ImageOps, ImageChops, ImageDraw
import numpy as np

import torch
import torch.utils.data as data
import torchvision
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
        data = np.asarray(im, dtype=np.int16)
        data += np.random.normal(scale=255 * self.stddev, size=data.shape).astype(np.int16)
        return Image.fromarray(data.clip(0, 255).astype(np.uint8))
    
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

class AutoCrop(object):
    def __call__(self, im):
        return im.crop(ImageOps.invert(im).getbbox())

class RandomBackground(object):
    def __init__(self, max):
        self.max = max

    def __call__(self, im):
        back = Image.new('L', im.size)
        ImageDraw.ImageDraw(back).rectangle([(0, 0), im.size], fill=255 - np.random.randint(self.max * 255))
        return ImageChops.multiply(im, back.split()[0]).split()[0]

Loader = Compose((
    RandomCrop(300),
    RandomRotation(10, resample=Image.BILINEAR),
    CenterCrop(256),
    RandomHorizontalFlip(),
    Random90Rotation(),
    RandomResizedCrop(128, scale=(0.5, 1.0), ratio=(1., 1.)),
    RandomBackground(0.3),
))

def DownscalerUp(n):
    return Compose((
        lambda im: im.filter(ImageFilter.GaussianBlur(n)),
        GaussianNoise(0.08),
        Resize((64, 64), interpolation=Image.BILINEAR),
        Resize((128, 128), interpolation=Image.BILINEAR),
        ToTensor(),
    ))

def Downscaler(n):
    return Compose((
        lambda im: im.filter(ImageFilter.GaussianBlur(n)),
        GaussianNoise(0.08),
        Resize((64, 64), interpolation=Image.BILINEAR),
        ToTensor(),
    ))

def load_img(filepath):
    img = Image.open(filepath).convert('YCbCr')
    y, _, _ = img.split()
    return AutoCrop()(y)

def make_pair(image, reupscale):
    orig = Loader(image)
    hr = ToTensor()(orig)
    if reupscale:
        lr = DownscalerUp(np.random.choice([0.5, 1., 1.5, 2.]))(orig)
    else:
        lr = Downscaler(np.random.choice([0.5, 1., 1.5, 2.]))(orig)

    #if index == 0 and np.random.rand() < 0.01:
    #    orig.save('data_orig.png')
    #    save_image('data_hr.png', hr)
    #    save_image('data_lr.png', lr)

    return lr, hr

def fill_queue(queue, images, reupscale):
    while True:
        queue.put(make_pair(images[np.random.choice(len(images))], reupscale), block=True)

class DatasetFromFolder(data.Dataset):
    @timeit
    def __init__(self, image_dir, reupscale=False, decimate=None):
        super(DatasetFromFolder, self).__init__()
        self.reupscale = reupscale
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        if decimate is not None:
            self.image_filenames = [fn for fn in self.image_filenames if np.random.rand() < decimate]

        with Executor() as e:
            self.images = list(e.map(load_img, self.image_filenames))

        self.queue = Queue(maxsize=256)
        self.threads = [threading.Thread(target=fill_queue, args=(self.queue, self.images, reupscale)) for _ in range(4)]
        for t in self.threads:
            t.start()

        # self.executor = Executor()
        # self.queue = Queue(maxsize=128)
        # self.requeue()

    def requeue(self):
        if self.queue.qsize() <= self.queue.maxsize // 2:
            for _ in range(self.queue.maxsize - self.queue.qsize()):
                self.queue.put(self.executor.submit(self.make_pair, self.random_image()))

    def random_image(self):
        return np.random.choice(len(self.images))

    def make_pair(self, image_idx):
        return make_pair(self.images[image_idx], self.reupscale)

    def get(self, block=False):
        return self.queue.get(timeout=None if block else 1.0, block=block)
  
    def __getitem__(self, index):
        return self.queue.get(timeout=1.0)
        return self.make_pair(self.random_image())
        result = self.queue.get()
        self.requeue()
        if not result.done():
            print('delay!!')
        return result.result()

    def __len__(self):
        return 512

def dim3(tensor):
    return tensor.view(-1, tensor.size()[-2], tensor.size()[-1])

def batch_queue(queue, dataset, batch_size):
    while True:
        samples = [dataset.get(block=True) for _ in range(batch_size)]
        samples_lr, samples_hr = [dim3(lr) for lr, hr in samples], [dim3(hr) for lr, hr in samples]
        batch = torch.stack(samples_lr), torch.stack(samples_hr)
        queue.put(batch)

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

class Batcher(object):
    def __init__(self, dataset, big_batch, mini_batch):
        self.dataset = dataset
        self.queue = Queue(maxsize=3)
        self.big_batch = big_batch
        self.mini_batch = mini_batch

        self.threads = [threading.Thread(target=batch_queue, args=(self.queue, dataset, big_batch)) for _ in range(4)]
        for t in self.threads:
            t.start()

    def get(self):
        batch = self.queue.get(timeout=1.0)
        chunked = chunks(batch, self.mini_batch)
        for chunk in chunked:
            yield chunk

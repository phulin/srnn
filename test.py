#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun May  7 10:07:14 2017

@author: ldy
"""

from __future__ import print_function
from os.path import join
import argparse
import torch
from torch.autograd import Variable
from PIL import Image
from torchvision.transforms import ToTensor, Compose, CenterCrop
import numpy as np
from ssim import SSIM, MSSSIM
from train import timeit

# Training settings
parser = argparse.ArgumentParser(description='PyTorch LapSRN')
parser.add_argument('--model', type=str, default=join('model', 'model_epoch_50.pth'), help='model file to use')
parser.add_argument('--input', type=str, help='input image')
parser.add_argument('--reference', type=str, help='reference image')
parser.add_argument('--output', type=str, default='out.png', help='where to save the output image')
parser.add_argument('--cuda', action='store_true', help='use cuda')

opt = parser.parse_args()

print(opt)

def CharbonnierLoss(predict, target):
    return torch.mean(torch.sqrt(torch.pow((predict-target), 2) + 1e-3)) # epsilon=1e-3

def process(out):
    data = out.data
    print(data.size())
    if len(data.size()) == 4: data = data[0]
    if len(data.size()) == 3: data = data[0]
    out_img_y = data.numpy()
    print(out_img_y.min(), out_img_y.max())
    # out_img_y -= np.percentile(out_img_y, 10)
    # out_img_y /= np.percentile(out_img_y, 90)
    out_img_y *= 255.0
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = Image.fromarray(np.uint8(out_img_y), mode='L')

    return out_img_y

def average(images):
    w, h = images[0].size
    result = np.zeros((h, w), dtype=np.uint32)
    for im in images:
        result += np.asarray(im, dtype=np.uint8)
    result //= len(images)
    return Image.fromarray(result.clip(0, 255).astype(np.uint8))

model = torch.load(opt.model, map_location=lambda storage, loc: storage)
if opt.cuda:
    model = model.cuda()
    
transform = Compose((CenterCrop(2048),))

def run(y):
    LR = Variable(ToTensor()(y).unsqueeze(0).pin_memory(), volatile=True)
    if opt.cuda:
        LR = LR.cuda()
    HR_2 = timeit(model.__call__)(LR).cpu()
    if opt.reference:
        img = Image.open(opt.reference).convert('YCbCr')
        y, _, _ = img.split()
        # y = y.resize((y.size[0] // 2, y.size[1] // 2), Image.BILINEAR)
        y = transform(y)
        y.save('ref.png')
        HR_ref = Variable(ToTensor()(y).unsqueeze(0), volatile=True)
        print('SSIM:', 1 - SSIM()(HR_ref, HR_2).data[0])
        print('MS-SSIM:', 1 - MSSSIM()(HR_ref, HR_2).data[0])
        print('Charbonnier:', CharbonnierLoss(HR_ref, HR_2).data[0])
    return process(HR_2.cpu())

def Rotate(angle):
    return lambda im: im.rotate(angle)

def Transpose(spec):
    return lambda im: im.transpose(spec)

TRANSFORMS = [
    Rotate(0),
    Rotate(90),
    Rotate(180),
    Rotate(270),
    Transpose(Image.FLIP_LEFT_RIGHT),
    Transpose(Image.FLIP_TOP_BOTTOM),
    Transpose(Image.TRANSPOSE),
    Transpose(Image.TRANSVERSE),
]

INVERSES = [
    Rotate(0),
    Rotate(270),
    Rotate(180),
    Rotate(90),
    Transpose(Image.FLIP_LEFT_RIGHT),
    Transpose(Image.FLIP_TOP_BOTTOM),
    Transpose(Image.TRANSPOSE),
    Transpose(Image.TRANSVERSE),
]

# TRANSFORMS = ROTATIONS + [lambda im: Flip()(Rotate(90)(im)), Compose((Flip(), Rotate(180))), Compose((Flip(), Rotate(270)))]
# INVERSES = ROTATIONS_R + [lambda im: Rotate(270)(Flip()(im))]

# TRANSFORMS = [Compose((Rotate(90), Flip()))]

img = Image.open(opt.input).convert('YCbCr')
y, _, _ = img.split()
# y = transform(y)
y.save('in.png')
ys = [f(y) for f in TRANSFORMS]
ys_out = [run(x) for x in ys]
xs = [f(x) for f, x in zip(INVERSES, ys_out)]
average(xs).save(opt.output)
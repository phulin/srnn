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
from torchvision.transforms import ToTensor
import numpy as np

# Training settings
parser = argparse.ArgumentParser(description='PyTorch LapSRN')
parser.add_argument('--model', type=str, default=join('model', 'model_epoch_50.pth'), help='model file to use')
parser.add_argument('--input', type=str, help='input image')
parser.add_argument('--output', type=str, default='out.png', help='where to save the output image')
parser.add_argument('--cuda', action='store_true', help='use cuda')

opt = parser.parse_args()

print(opt)

def process(out):
    out_img_y = out.data[0].numpy()
    out_img_y *= 255.0
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')

    return out_img_y

model = torch.load(opt.model, map_location=lambda storage, loc: storage)
if opt.cuda:
    model = model.cuda()

img = Image.open(opt.input).convert('YCbCr')
y, _, _ = img.split()
LR = Variable(ToTensor()(y), volatile=True).view(1, -1, y.size[1], y.size[0])
if opt.cuda:
    LR = LR.cuda()
HR_2 = model(LR)
HR_2_out = process(HR_2.cpu())
HR_2_out.save(opt.output)
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

model = torch.load(opt.model, map_location=lambda storage, loc: storage)
if opt.cuda:
    model = model.cuda()
    
transform = Compose((CenterCrop(512),))

img = Image.open(opt.input).convert('YCbCr')
y, _, _ = img.split()
y = transform(y)
y.save('in.png')
LR = Variable(ToTensor()(y).unsqueeze(0), volatile=True)
if opt.cuda:
    LR = LR.cuda()
HR_2 = model(LR).cpu()
if opt.reference:
    img = Image.open(opt.reference).convert('YCbCr')
    y, _, _ = img.split()
    y = y.resize((y.size[0] // 2, y.size[1] // 2), Image.BILINEAR)
    y = transform(y)
    y.save('ref.png')
    HR_ref = Variable(ToTensor()(y).unsqueeze(0), volatile=True)
    print('SSIM:', 1 - SSIM()(HR_ref, HR_2).data[0])
    print('MS-SSIM:', 1 - MSSSIM()(HR_ref, HR_2).data[0])
    print('Charbonnier:', CharbonnierLoss(HR_ref, HR_2).data[0])
HR_2_out = process(HR_2.cpu())
HR_2_out.save(opt.output)
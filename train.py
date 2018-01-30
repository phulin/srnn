#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May  6 18:18:37 2017

@author: ldy
"""

from __future__ import print_function
import argparse
from os.path import exists, join
from os import makedirs

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model import LasSRN
from data import get_training_set

# Training settings 
parser = argparse.ArgumentParser(description='PyTorch LapSRN')
parser.add_argument('--batchSize', type=int, default=64, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=1e-5, help='Learning Rate.')
parser.add_argument('--weightDecay', type=float, default=1e-4, help='Weight decay.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--checkpoint', type=str, default='model', help='Path to checkpoint')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
opt = parser.parse_args()

print(opt)

cuda = opt.cuda
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
train_set = get_training_set()
training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=opt.batchSize, shuffle=False)


def CharbonnierLoss(predict, target):
    return torch.mean(torch.sqrt(torch.pow((predict-target), 2) + 1e-6)) # epsilon=1e-3


print('===> Building model')
model = LasSRN()
model_out_path = join('model', "model_epoch_{}.pth".format(0))
torch.save(model, model_out_path)
#criterion = CharbonnierLoss()
criterion = nn.MSELoss()
print (model)
if cuda:
    model = model.cuda()
    criterion = criterion.cuda()

def train(epoch, i):
    epoch_loss = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        LR, HR_2_target = Variable(batch[0]), Variable(batch[1])
        
        if cuda:
            LR = LR.cuda()
            # HR_2_target = HR_2_target.cuda()

        optimizer.zero_grad()
        HR_2 = model(LR).cpu()

        loss = CharbonnierLoss(HR_2, HR_2_target)

        epoch_loss += loss.data[0]
        loss.backward()
        optimizer.step()

        # print("===> Epoch[{}], Loop{}({}/{}): Loss: {:.4f}".format(epoch, i, iteration, len(training_data_loader), loss.data[0]))
    print("===> Epoch[{}], Loop {}: Avg. Loss: {:.4f}".format(epoch, i, epoch_loss / len(training_data_loader)))

def checkpoint(epoch):
    if not exists(opt.checkpoint):
        makedirs(opt.checkpoint)
    model_out_path = join('model', "model_epoch_{}.pth".format(epoch))
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

lr=opt.lr
for epoch in range(1, opt.nEpochs + 1):

    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weightDecay)
    for i in range(1000):
        train(epoch, i)
    checkpoint(epoch)
    if epoch % 50 == 0:
        lr /= 2

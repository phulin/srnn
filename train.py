#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May  6 18:18:37 2017

@author: ldy
"""

from __future__ import print_function
import argparse
import json
from os.path import exists, join
from os import listdir, makedirs
import re

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from model import LasSRN
from data import get_training_set
from dataloader import DataLoader

def train(epoch, i):
    epoch_loss = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        LR, HR_2_target = Variable(batch[0]), Variable(batch[1])
        
        if cuda:
            LR = LR.cuda()
            HR_2_target = HR_2_target.cuda()

        optimizer.zero_grad()
        HR_2 = model(LR)

        loss = CharbonnierLoss(HR_2, HR_2_target)

        epoch_loss += loss.data[0]
        loss.backward()
        optimizer.step()

        # print("===> Epoch[{}], Loop{}({}/{}): Loss: {:.4f}".format(epoch, i, iteration, len(training_data_loader), loss.data[0]))
    print("===> Epoch[{}], Loop {}: Avg. Loss: {:.4f}".format(epoch, i, epoch_loss / len(training_data_loader)))

def checkpoint(model, name):
    if not exists(opt.checkpoint):
        makedirs(opt.checkpoint)
    model_out_path = join(opt.checkpoint, name)
    torch.save(model, model_out_path)
    print("**** Checkpoint saved to {}".format(model_out_path))

def CharbonnierLoss(predict, target):
    return torch.mean(torch.sqrt(torch.pow((predict-target), 2) + 1e-6)) # epsilon=1e-3

if __name__ == '__main__':
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
    training_data_loader = DataLoader(dataset=train_set, batch_size=opt.batchSize,
                                      pin_memory=True, num_workers=4)

    model = None
    epoch0 = 1
    loop0 = 1
    if exists(join(opt.checkpoint, 'model_latest.pth')) \
            and exists(join(opt.checkpoint, 'model_latest.json')):
        print('===> Loading model from inter-epoch file.')
        model = torch.load(join(opt.checkpoint, 'model_latest.pth'))
        with open(join(opt.checkpoint, 'model_latest.json')) as f:
            params = json.load(f)
            epoch0 = params['epoch']
            loop0 = params['loop']
    else:
        print('===> Loading model from epoch checkpoint.')
        checkpoints = listdir(opt.checkpoint)
        matches = [re.match(r'model_epoch_([1-9][0-9]*).pth', c) for c in checkpoints]
        epochs = [int(match.group(1)) for match in matches if match is not None]
        if epochs:
            epoch0 = max(epochs)
            model_path = join(opt.checkpoint, 'model_epoch_{}.pth'.format(epoch0))
            model = torch.load(model_path)
            print('===> Loaded model at {}.'.format(model_path))

    if model is None:
        print('===> Building model from scratch.')
        model = LasSRN()
        checkpoint(model, 'model_epoch_0.pth')
        print (model)
        if cuda:
            model = model.cuda()
    
    for epoch in range(epoch0, opt.nEpochs + 1):
        optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weightDecay)

        for i in range(loop0, 1001) if epoch == epoch0 else range(1, 1001):
            train(epoch, i)
            if i > 0 and i < 1000 and i % 50 == 0:
                checkpoint(model, "model_latest.pth")
                with open(join(opt.checkpoint, "model_latest.json"), 'w') as f:
                    json.dump({ 'epoch': epoch, 'loop': i }, f)
    
        if epoch % 5 == 0:
            checkpoint(model, 'model_epoch_{}.pth'.format(epoch))
        if epoch % 50 == 0:
            lr /= 2

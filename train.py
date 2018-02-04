# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
import json
import numpy as np
from PIL import Image
from os.path import exists, isdir, join
from os import listdir, makedirs
import re
import time

import torch
import torch.optim as optim
from torch.autograd import Variable
import torchvision.utils
import torchvision.transforms as transforms

from data import get_training_set
from torch.utils.data import DataLoader
#from dataloader import DataLoader
from model import MSLapSRN


def count_parameters(model):
    total_param = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_param = np.prod(param.size())
            if param.dim() > 1:
                print(name, ':', 'x'.join(str(x) for x in list(param.size())), '=', num_param)
            else:
                print(name, ':', num_param)
            total_param += num_param
    return total_param

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print('*** timer:%r  %2.2f ms' % \
                (method.__name__, (te - ts) * 1000))
        return result
    return timed

def save_tensor(name, out, scale=1.0, **kwargs):
    grid = torchvision.utils.make_grid(out.data, range=(0., 1.), **kwargs)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    image = Image.fromarray(ndarr)
    if abs(scale - 1.0) > 1e-4:
        w, h = image.size
        new_w, new_h = int(scale * w), int(scale * h)
        image = image.resize((new_w, new_h), resample=Image.BICUBIC)
    image.save(name)

#@timeit
def train_loop(epoch, i):
    total_loss = 0.
    for iteration, batch in enumerate(training_data_loader, 1):
        LR, HR_2_target = Variable(batch[0]), Variable(batch[1])

        if cuda:
            LR = LR.cuda()
            HR_2_target = HR_2_target.cuda()

        optimizer.zero_grad()
        HR_2, = model(LR)

        if iteration == 1 and i % 50 == 0:
            save_tensor('lr.png', LR.cpu())
            save_tensor('lr_bicubic.png', LR.cpu(), scale=2.0)
            save_tensor('hr_target.png', HR_2_target.cpu())
            save_tensor('hr_modeled.png', HR_2.cpu())
        loss = CharbonnierLoss(HR_2, HR_2_target)

        total_loss += loss.data[0]
        loss.backward()
        optimizer.step()

        # print("===> Epoch[{}], Loop{}({}/{}): Loss: {:.4f}".format(epoch, i, iteration, len(training_data_loader), loss.data[0]))

    return total_loss / len(training_data_loader)

def train_epoch(epoch, loop0):
    total_loss = 0.
    loops_loss = 0.
    loops_count = 0
    for i in range(loop0, 1001):
        loop_loss = train_loop(epoch, i)
        total_loss += loop_loss
        loops_loss += loop_loss
        loops_count += 1

        if i % 10 == 0:
            print("===> Epoch[{}], Loop {}: Avg. Loss: {:.4f}".format(epoch, i, loops_loss / loops_count))
            loops_loss = 0.
            loops_count = 0
    
        if i > 0 and i < 1000 and i % 50 == 0:
            checkpoint(model, "model_latest.pth")
            with open(join(opt.checkpoint, "model_latest.json"), 'w') as f:
                json.dump({ 'epoch': epoch, 'loop': i }, f)

    print("=> Epoch[{}]: Avg. Loss: {:.4f}".format(epoch, total_loss / (1001 - loop0)))
    return total_loss

def checkpoint(model, name):
    if not exists(opt.checkpoint):
        makedirs(opt.checkpoint)
    model_out_path = join(opt.checkpoint, name)
    torch.save(model, model_out_path)
    print("**** Checkpoint saved to {}".format(model_out_path))

def CharbonnierLoss(predict, target):
    return torch.mean(torch.sqrt(torch.pow((predict-target), 2) + 1e-3)) # epsilon=1e-3

if __name__ == '__main__':
    # Training settings 
    parser = argparse.ArgumentParser(description='PyTorch LapSRN')
    parser.add_argument('--batchSize', type=int, default=24, help='training batch size')
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
                                      pin_memory=True, num_workers=0)

    model = None
    epoch0 = 1
    loop0 = 1
    lr = opt.lr
    if isdir(opt.checkpoint):
        if exists(join(opt.checkpoint, 'model_latest.pth')) \
                and exists(join(opt.checkpoint, 'model_latest.json')):
            print('===> Loading model from inter-epoch file.')
            model = torch.load(join(opt.checkpoint, 'model_latest.pth'))
            with open(join(opt.checkpoint, 'model_latest.json')) as f:
                params = json.load(f)
                epoch0 = params['epoch']
                loop0 = params['loop']
                lr /= 2 ** (epoch0 // 100)
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
        model = MSLapSRN(depth=5, recursive_blocks=8, levels=1)
        print('===> Number of params:', count_parameters(model))
        checkpoint(model, 'model_epoch_0.pth')
        print(model)

        if cuda:
            print('===> Moving model to GPU.')
            model = model.cuda()

    for epoch in range(epoch0, opt.nEpochs + 1):
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=opt.momentum, weight_decay=opt.weightDecay)

        if epoch > epoch0:
            loop0 = 1

        train_epoch(epoch, loop0)

        if epoch % 5 == 0:
            checkpoint(model, 'model_epoch_{}.pth'.format(epoch))
        if epoch % 100 == 0:
            lr /= 2

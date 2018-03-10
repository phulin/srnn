# -*- coding: utf-8 -*-

from __future__ import print_function
# from dataset import Batcher

import argparse
import json
import numpy as np
from PIL import Image
from os.path import exists, isdir, join
from os import listdir, makedirs
import re
import time

import multiprocessing as mp

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision.utils
from torchvision.models import vgg19

from torch.utils.data import DataLoader

from dataset import DatasetFromFolder
#from dataloader import DataLoader
from srcnn import CTSRCNN
from edsr import EDSR
from ssim import SSIM, MSSSIM

def count_parameters(model):
    total_param = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            total_param += np.prod(param.size())
            # print('{}: {} = {}'.format(name, param.size(), np.prod(param.size())))
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

class Loss(object):
    def __add__(self, other):
        return AddLoss(self, other)

    def __mul__(self, other):
        return MulLoss(self, other)

    def cuda(self):
        return self

class ConstLoss(Loss):
    def __init__(self, a):
        self.a = a

    def __call__(self, x, y):
        return self.a

def make_loss(a):
    if hasattr(a, '__call__'):
        return a
    else:
        return ConstLoss(a)

class AddLoss(Loss):
    def __init__(self, a, b):
        self.a = make_loss(a)
        self.b = make_loss(b)

    def __call__(self, x, y):
        return self.a(x, y) + self.b(x, y)

    def cuda(self):
        return AddLoss(self.a.cuda(), self.b.cuda())

class MulLoss(Loss):
    def __init__(self, a, b):
        self.a = make_loss(a)
        self.b = make_loss(b)

    def __call__(self, x, y):
        return self.a(x, y) * self.b(x, y)

    def cuda(self):
        return MulLoss(self.a.cuda(), self.b.cuda())

class CharbonnierLoss(Loss):
    def __init__(self, eps=1e-3):
        self.eps = eps

    def __call__(x, y):
        return torch.mean(torch.sqrt(torch.pow(x - y, 2) + self.eps))

class SSIMLoss(Loss):
    def __init__(self, *args, **kwargs):
        self.ssim = SSIM(*args, **kwargs)

    def __call__(self, x, y):
        return 1 - self.ssim(x, y)

class MSSSIMLoss(Loss):
    def __init__(self, *args, **kwargs):
        self.ssim = MSSSIM(*args, **kwargs)

    def __call__(self, x, y):
        return 1 - self.ssim(x, y)

def broadcast_color(tensor):
    return torch.cat((tensor, tensor, tensor), dim=1)

class VGG19Loss(Loss):
    def __init__(self, i, j):
        vgg = vgg19(pretrained=True)

        i_current, j_current = 1, 0
        self.k = None
        for k, layer in enumerate(vgg.features):
            if isinstance(layer, nn.Conv2d):
                j_current += 1
            elif isinstance(layer, nn.MaxPool2d):
                i_current += 1
                j_current = 0
            elif isinstance(layer, nn.ReLU) and (i, j) == (i_current, j_current):
                self.k = k
                break

        assert self.k is not None

        self.model = nn.Sequential(*[l for l in vgg.features][:k + 1])

    def cuda(self):
        self.model = self.model.cuda()
        return self

    def __call__(self, x, y):
        if x.size()[1] == 1: x = broadcast_color(x)
        if y.size()[1] == 1: y = broadcast_color(y)
        return torch.mean(torch.pow(self.model(x) - self.model(y), 2))

class Trainer(object):
    LOSSES = {
        'charbonnier': CharbonnierLoss,
        'mse': torch.nn.MSELoss,
        'ssim': SSIMLoss(),
        'ssim_char': SSIMLoss() * 0.84 + CharbonnierLoss() * 0.16,
        'msssim': MSSSIMLoss(),
        'msssim_char': MSSSIMLoss() * 0.6 + CharbonnierLoss() * 0.4,
        'vgg19': VGG19Loss(5, 4) * 0.5 + VGG19Loss(2, 2) * 0.2,
    }

    TYPE_KWARGS = {
        CTSRCNN: { 'trim': True, 'add_layers': True, 'reupscale': True },
        EDSR: { 'add_layers': True },
    }

    N_LOOPS = 1000
    DISPLAY_INTERVAL = 5
    SAVE_INTERVAL = 100
    RUNNING_LEN = 400

    TOLERANCE = 0.015
    TRIM_TOLERANCE = 0.01
    FINAL_TOLERANCE = -np.inf

    def __init__(self, model, epoch=0, loop=0, current_epoch_loss=0., last_epoch_loss=None, loss=None, checkpoint_dir=None, optimizer='sgd'):
        self.model = model
        self.epoch = epoch
        self.loop = loop
        self.current_epoch_loss = loop * current_epoch_loss
        self.last_epoch_loss = Trainer.N_LOOPS * last_epoch_loss if last_epoch_loss is not None else None
        self.loss = loss
        self.loss_fn = Trainer.LOSSES[loss]
        if cuda:
            self.loss_fn = self.loss_fn.cuda()
        self.checkpoint_dir = checkpoint_dir
        self.running_array = np.zeros((Trainer.RUNNING_LEN,), dtype=np.float64)
        self.last_display = None

        if optimizer == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weightDecay)
        elif optimizer == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=opt.lr, betas=(0.9, 0.999), weight_decay=opt.weightDecay)

        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, opt.lr_period, gamma=0.5)

        self.add_layers = False
        self.trim = False
        self.reupscale = False
        if type(model) in Trainer.TYPE_KWARGS:
            for key, val in Trainer.TYPE_KWARGS[type(model)].items():
                setattr(self, key, val)

        print('===> Loading datasets')
        self.dataset = DatasetFromFolder(opt.train, reupscale=self.reupscale,
                                    decimate=0.05 if opt.decimate else None)
        self.batch_size = opt.batchSize

    @staticmethod
    def restore(input_dir, checkpoint_dir=None, **kwargs):
        if checkpoint_dir is None:
            checkpoint_dir = input_dir
        elif input_dir is None:
            input_dir = checkpoint_dir

        if not isdir(input_dir):
            print('===> No checkpoint found in {}'.format(input_dir))
            return None

        if exists(join(input_dir, 'model_latest.pth')) \
                and exists(join(input_dir, 'model_latest.json')):
            print('===> Loading model from inter-epoch file.')
            model = torch.load(join(input_dir, 'model_latest.pth'))
            if opt.cuda: model = model.cuda()
            with open(join(input_dir, 'model_latest.json')) as f:
                params = json.load(f)
                kwargs.update(params)
                return Trainer(model, checkpoint_dir=checkpoint_dir, **kwargs)
        else:
            print('===> Loading model from epoch checkpoint.')
            checkpoints = listdir(input_dir)
            matches = [re.match(r'model_epoch_([1-9][0-9]*).pth', c) for c in checkpoints]
            epochs = [int(match.group(1)) for match in matches if match is not None]
            if epochs:
                epoch0 = max(epochs)
                model_path = join(input_dir, 'model_epoch_{}.pth'.format(epoch0))
                model = torch.load(model_path)
                print('===> Loaded model at {}.'.format(model_path))
                return Trainer(model, epoch0, checkpoint_dir=checkpoint_dir, **kwargs)

        return None

    def checkpoint(self, name):
        if not exists(self.checkpoint_dir):
            makedirs(self.checkpoint_dir)
        model_out_path = join(self.checkpoint_dir, name)
        torch.save(self.model, model_out_path)
        print("**** Checkpoint saved to {}".format(model_out_path))

    def save(self):
        self.checkpoint("model_latest.pth")
        with open(join(self.checkpoint_dir, "model_latest.json"), 'w') as f:
            global last_epoch_loss
            json.dump({
                'epoch': self.epoch,
                'loop': self.loop,
                'current_epoch_loss': self.current_epoch_loss / self.loop,
                'last_epoch_loss': self.last_epoch_loss / Trainer.N_LOOPS if self.last_epoch_loss is not None else None,
            }, f)

    #@timeit
    def train_loop(self, loader):
        loop_loss = 0.
        for iteration, batch in enumerate(loader, 1):
            LR, HR_2_target = Variable(batch[0]), Variable(batch[1])

            if cuda:
                HR_2_target = HR_2_target.cuda()
                LR = LR.cuda()

            self.optimizer.zero_grad()
            HR_2 = self.model(LR)

            if iteration == 1 and self.loop % 100 == 0:
                save_tensor('lr_orig.png', LR.cpu())
                save_tensor('lr_bicubic.png', LR.cpu(), scale=2.0)
                save_tensor('hr_target.png', HR_2_target.cpu())
                save_tensor('hr_modeled.png', HR_2.cpu())

            loss = self.loss_fn(HR_2, HR_2_target)
            loop_loss += loss.data[0]

            loss.backward()
            self.optimizer.step()

            # print("===> Epoch[{}], Loop{}({}/{}): Loss: {:.4f}".format(epoch, i, iteration, len(training_data_loader), loss.data[0]))

        return loop_loss / iteration

    def train_epoch(self, loader):
        loop0 = self.loop if self.loop < Trainer.N_LOOPS else 0
        for loop in range(loop0 + 1, Trainer.N_LOOPS + 1):
            self.loop = loop

            loop_loss = self.train_loop(loader)

            running_idx = loop % self.running_array.shape[0]
            self.running_array[running_idx] = loop_loss

            self.current_epoch_loss += loop_loss

            if loop % Trainer.DISPLAY_INTERVAL == 0:
                running_avg = self.running_array.sum() / np.count_nonzero(self.running_array)
                loops_loss = np.roll(self.running_array, -running_idx - 1)[-Trainer.DISPLAY_INTERVAL:]
                loops_avg = loops_loss.sum() / np.count_nonzero(loops_loss)

                status = "===> Epoch[{}], Loop {:4d}: Avg. Loss: {:.4f}, Running {}: {:.4f}, Epoch: {:.5f}".format(
                    self.epoch, loop, loops_avg,
                    self.running_array.shape[0], running_avg,
                    self.current_epoch_loss / loop,
                )

                if self.last_epoch_loss is not None:
                    tolerance = Trainer.TOLERANCE if self.model.depth() < opt.maxDepth \
                        else Trainer.TRIM_TOLERANCE if self.trim and self.model.trim_count() < opt.maxTrim \
                        else Trainer.FINAL_TOLERANCE
                    status += ", Target: {:.5f}".format((1 - tolerance) * self.last_epoch_loss / Trainer.N_LOOPS)

                if self.last_display is not None:
                    status += "; {:2.2f}s".format((time.time() - self.last_display) / Trainer.DISPLAY_INTERVAL)

                print(status)

                self.last_display = time.time()

            if loop > loop0 and loop < Trainer.N_LOOPS and loop % Trainer.SAVE_INTERVAL == 0:
                self.save()

        print("=> Epoch[{}]: Avg. Loss: {:.4f}".format(self.epoch, self.current_epoch_loss / Trainer.N_LOOPS))
        return self.current_epoch_loss

    def train(self):
        epoch0 = self.epoch

        while True:
            try:
                # loader = Batcher(train_set, big_batch=64, mini_batch=opt.batchSize)
                loader = DataLoader(self.dataset, batch_size=self.batch_size,
                                    pin_memory=cuda,
                                    num_workers=mp.cpu_count())
                self.train_loop(loader)
                break
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    self.batch_size = int(self.batch_size * 0.9)
                    print('===> Reducing batch size to {}.'.format(self.batch_size))
                    if self.batch_size == 0:
                        print("can't reduce batchSize any further...")
                        raise
                else:
                    raise


        self.scheduler.step(epoch0)
        for epoch in range(epoch0, opt.nEpochs + 1):
            self.epoch = epoch

            self.current_epoch_loss = 0.
            self.scheduler.step()
            epoch_loss = self.train_epoch(loader)

            if self.last_epoch_loss is not None:
                relative_change = (self.last_epoch_loss - epoch_loss) / self.last_epoch_loss
                depth = self.model.depth()
                print('=> Relative change: {:.1%}. Current depth: {}.'.format(relative_change, depth))

                adding_done = not self.add_layers or depth >= opt.maxDepth
                trimming_done = not self.trim or depth >= opt.maxTrim

                if adding_done and trimming_done and relative_change < Trainer.FINAL_TOLERANCE:
                    print('**** STOPPING ****')
                    self.checkpoint('model_epoch_{}_final.pth'.format(epoch))
                    break
                elif not adding_done and relative_change < Trainer.TOLERANCE and relative_change >= -0.5:
                    print('**** ADDING 2 MORE LAYERS ****')
                    self.checkpoint('model_epoch_{}_depth_{}.pth'.format(epoch, depth))
                    self.model.add_layers()
                    print('**** CURRENT DEPTH: {} ****'.format(self.model.depth()))
                    print('===> Number of params:', count_parameters(trainer.model))
                    self.last_epoch_loss = None
                elif not trimming_done and relative_change < Trainer.TRIM_TOLERANCE and relative_change >= -0.5:
                    print('**** TRIMMING 2 LAYERS ****')
                    self.checkpoint('model_epoch_{}_trim_{}.pth'.format(epoch, self.model.trim_count()))
                    self.model.trim()
                    print('**** CURRENT TRIM: {} ****'.format(self.model.trim_count()))
                    print('===> Number of params:', count_parameters(trainer.model))
                    self.last_epoch_loss = None
                else:
                    self.last_epoch_loss = epoch_loss
            else:
                self.last_epoch_loss = epoch_loss

            if epoch % 5 == 0:
                self.checkpoint('model_epoch_{}.pth'.format(epoch))

if __name__ == '__main__':

    MODELS = {
        'ct-srcnn': CTSRCNN,
        'edsr': EDSR,
    }

    # Training settings 
    parser = argparse.ArgumentParser(description='PyTorch LapSRN')
    parser.add_argument('--batchSize', type=int, default=32, help='training batch size')
    parser.add_argument('--nEpochs', type=int, default=250, help='number of epochs to train for')
    parser.add_argument('--optimizer', type=str, choices=['sgd', 'adam'], default='sgd', help='Optimizer type.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate.')
    parser.add_argument('--lr_period', type=int, default=30, help='Decay period for LR.')
    parser.add_argument('--weightDecay', type=float, default=0, help='Weight decay.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--loss', type=str, choices=Trainer.LOSSES.keys(), default='ssim', help='Loss function.')
    parser.add_argument('--type', type=str, choices=MODELS.keys(), default='edsr', help='Model type to build.')
    parser.add_argument('--start', type=str, default=None, help='Path to starting point')
    parser.add_argument('--checkpoint', type=str, default='model', help='Path to checkpoint')
    parser.add_argument('--cuda', action='store_true', help='use cuda?')
    parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
    parser.add_argument('--train', type=str, default='/train/hi', help='training examples')
    parser.add_argument('--decimate', action='store_true', help='fast startup')
    parser.add_argument('--maxDepth', type=int, default=21, help='Max depth to grow network to')
    parser.add_argument('--maxTrim', type=int, default=16, help='Max depth to trim network to')
    opt = parser.parse_args()

    print(opt)

    cuda = opt.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    trainer = Trainer.restore(opt.start, checkpoint_dir=opt.checkpoint, loss=opt.loss)

    if trainer is None:
        print('===> Building model from scratch.')
        model_class = MODELS[opt.type]
        model = model_class()

        if cuda:
            print('===> Moving model to GPU.')
            model = model.cuda()

        trainer = Trainer(model, checkpoint_dir=opt.checkpoint, loss=opt.loss)
        trainer.checkpoint('model_epoch_0.pth')

    print(trainer.model)
    print('===> Number of params:', count_parameters(trainer.model))

    trainer.train()

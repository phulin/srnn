# -*- coding: utf-8 -*-

from __future__ import print_function
from dataset import Batcher

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

def CharbonnierLoss(predict, target):
    return torch.mean(torch.sqrt(torch.pow((predict-target), 2) + 1e-3)) # epsilon=1e-3

class SSIMLoss(object):
    def __init__(self, *args, **kwargs):
        self.ssim = SSIM(*args, **kwargs)

    def __call__(self, x, y):
        return 1 - self.ssim(x, y)

class MSSSIMLoss(object):
    def __init__(self, *args, **kwargs):
        self.ssim = MSSSIM(*args, **kwargs)

    def __call__(self, x, y):
        return 1 - self.ssim(x, y)

class SSIM_CharbonnierLoss(object):
    def __init__(self, alpha=0.84, *args, **kwargs):
        self.ssim = SSIM(*args, **kwargs)
        self.alpha = alpha

    def __call__(self, x, y):
        return self.alpha * (1 - self.ssim(x, y)) + (1 - self.alpha) * CharbonnierLoss(x, y)

class MSSSIM_CharbonnierLoss(object):
    def __init__(self, alpha=0.84, *args, **kwargs):
        self.msssim = MSSSIM(*args, **kwargs)
        self.alpha = alpha

    def __call__(self, x, y):
        return self.alpha * (1 - self.msssim(x, y)) + (1 - self.alpha) * CharbonnierLoss(x, y)

class Trainer(object):
    LOSSES = {
        'charbonnier': CharbonnierLoss,
        'mse': torch.nn.MSELoss,
        'ssim': SSIMLoss(),
        'ssim_char': SSIM_CharbonnierLoss(),
        'msssim': MSSSIMLoss(),
        'msssim_char': MSSSIM_CharbonnierLoss(alpha=0.6),
    }

    TYPE_KWARGS = {
        CTSRCNN: { 'trim': True, 'add_layers': True },
        EDSR: { 'add_layers': True },
    }

    N_LOOPS = 4000
    DISPLAY_INTERVAL = 25
    SAVE_INTERVAL = 100
    RUNNING_LEN = 400

    MAX_DEPTH = 25
    MAX_TRIM = 20
    TOLERANCE = 0.015
    TRIM_TOLERANCE = 0.01
    FINAL_TOLERANCE = -np.inf

    def __init__(self, model, epoch=0, loop=0, current_epoch_loss=0., last_epoch_loss=None, loss=None, checkpoint_dir=None, loader=None, optimizer='sgd'):
        self.model = model
        self.epoch = epoch
        self.loop = loop
        self.current_epoch_loss = loop * current_epoch_loss
        self.last_epoch_loss = Trainer.N_LOOPS * last_epoch_loss if last_epoch_loss is not None else None
        self.loss = loss
        self.loss_fn = Trainer.LOSSES[loss]
        self.checkpoint_dir = checkpoint_dir
        self.loader = loader
        self.running_array = np.zeros((Trainer.RUNNING_LEN,), dtype=np.float64)
        self.last_display = None

        if optimizer == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weightDecay)
        elif optimizer == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=opt.lr, betas=(0.9, 0.999), weight_decay=opt.weightDecay)
        
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, opt.lr_period, gamma=0.5)
        self.scheduler.step(epoch)

        self.add_layers = False
        self.trim = False
        if type(model) in Trainer.TYPE_KWARGS:
            for key, val in Trainer.TYPE_KWARGS[type(model)].items():
                setattr(self, key, val)

    @staticmethod
    def restore(checkpoint_dir, **kwargs):
        if not isdir(checkpoint_dir): return None

        if exists(join(checkpoint_dir, 'model_latest.pth')) \
                and exists(join(checkpoint_dir, 'model_latest.json')):
            print('===> Loading model from inter-epoch file.')
            model = torch.load(join(checkpoint_dir, 'model_latest.pth'))
            with open(join(checkpoint_dir, 'model_latest.json')) as f:
                params = json.load(f)
                kwargs.update(params)
                return Trainer(model, checkpoint_dir=checkpoint_dir, **kwargs)
        else:
            print('===> Loading model from epoch checkpoint.')
            checkpoints = listdir(checkpoint_dir)
            matches = [re.match(r'model_epoch_([1-9][0-9]*).pth', c) for c in checkpoints]
            epochs = [int(match.group(1)) for match in matches if match is not None]
            if epochs:
                epoch0 = max(epochs)
                model_path = join(checkpoint_dir, 'model_epoch_{}.pth'.format(epoch0))
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
    def train_loop(self):
        loop_loss = 0.
        for iteration, batch in enumerate(self.loader, 1):
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

    def train_epoch(self):
        loop0 = self.loop if self.loop < Trainer.N_LOOPS else 0
        for loop in range(loop0 + 1, Trainer.N_LOOPS + 1):
            self.loop = loop

            loop_loss = self.train_loop()

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
                    tolerance = Trainer.TOLERANCE if self.model.depth() < Trainer.MAX_DEPTH \
                        else Trainer.TRIM_TOLERANCE if self.trim and self.model.trim_count() < Trainer.MAX_TRIM \
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
        
        for epoch in range(epoch0, opt.nEpochs + 1):
            self.epoch = epoch
            self.scheduler.step()
            epoch_loss = self.train_epoch()

            if self.last_epoch_loss is not None:
                relative_change = (self.last_epoch_loss - epoch_loss) / self.last_epoch_loss
                depth = self.model.depth()
                print('=> Relative change: {:.1%}. Current depth: {}.'.format(relative_change, depth))

                adding_done = not self.add_layers or depth >= Trainer.MAX_DEPTH
                trimming_done = not self.trim or depth >= Trainer.MAX_TRIM

                if adding_done and trimming_done and relative_change < Trainer.FINAL_TOLERANCE:
                    print('**** STOPPING ****')
                    self.checkpoint('model_epoch_{}_final.pth'.format(epoch))
                    break
                elif not adding_done and relative_change < Trainer.TOLERANCE:
                    print('**** ADDING 2 MORE LAYERS ****')
                    self.checkpoint('model_epoch_{}_depth_{}.pth'.format(epoch, depth))
                    self.model.add_layers()
                    print('**** CURRENT DEPTH: {} ****'.format(self.model.depth()))
                    print('===> Number of params:', count_parameters(trainer.model))
                    self.last_epoch_loss = None
                elif not trimming_done and relative_change < Trainer.TRIM_TOLERANCE:
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
            self.current_epoch_loss = 0.    

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
    parser.add_argument('--checkpoint', type=str, default='model', help='Path to checkpoint')
    parser.add_argument('--cuda', action='store_true', help='use cuda?')
    parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
    parser.add_argument('--train', type=str, default='/train/hi', help='training examples')
    opt = parser.parse_args()

    print(opt)

    cuda = opt.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    print('===> Loading datasets')
    train_set = DatasetFromFolder(opt.train, reupscale=True)
    # train_set = get_training_set(reupscale=True, decimate=.05)
    # loader = Batcher(train_set, big_batch=64, mini_batch=opt.batchSize)
    loader = DataLoader(train_set, batch_size=opt.batchSize, pin_memory=True)

    trainer = Trainer.restore(opt.checkpoint, loss=opt.loss, loader=loader)

    if trainer is None:
        print('===> Building model from scratch.')
        model_class = MODELS[opt.type]
        train_set.reupscale = model_class != EDSR
        model = model_class()

        if cuda:
            print('===> Moving model to GPU.')
            model = model.cuda()

        trainer = Trainer(model, loss=opt.loss, loader=training_data_loader, checkpoint_dir=opt.checkpoint)
        trainer.checkpoint('model_epoch_0.pth')

    print(trainer.model)
    print('===> Number of params:', count_parameters(trainer.model))

    trainer.train()

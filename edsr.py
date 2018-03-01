import torch
import torch.nn as nn
import numpy as np

MAPS = 64
RESIDUAL_SCALING = 0.1

RES_BLOCKS = 32

LRELU_SLOPE = 0.01

def Conv2dInit(*args, rectified=False, **kwargs):
    result = nn.Conv2d(*args, **kwargs)
    if rectified:
        nn.init.kaiming_normal(result.weight.data, a=LRELU_SLOPE)
    else:
        nn.init.xavier_normal(result.weight.data)
    result.bias.data.zero_()
    return result

def Block(maps):
    return nn.Sequential(
        Conv2dInit(maps, maps, (3, 3), padding=1),
        nn.LeakyReLU(negative_slope=LRELU_SLOPE),
        Conv2dInit(maps, maps, (3, 3), padding=1, rectified=True),
    )

class Residual(nn.Module):
    def __init__(self, modules, scale=RESIDUAL_SCALING):
        nn.Module.__init__(self)
        if isinstance(modules, nn.Module):
            self.body = modules
        else:
            assert isinstance(modules, (tuple, list))
            self.body = nn.Sequential(*modules)
        self.scale = scale

    def forward(self, x):
        if self.scale is not None:
            return x + self.scale * self.body(x)
        else:
            return x + self.body(x)

class Shift(nn.Module):
    def __init__(self, amount):
        nn.Module.__init__(self)
        self.amount = amount

    def forward(self, x):
        return x + self.amount

    def __repr__(self):
        return 'Shift({})'.format(self.amount)

class EDSR(nn.Module):
    def __init__(self, res_blocks=6, factor=2, maps=MAPS):
        nn.Module.__init__(self)

        self.maps = maps
        res_blocks = [Residual(Block(maps)) for _ in range(res_blocks)]

        self.head = nn.Sequential(
            Conv2dInit(1, maps, (3, 3), padding=1),
            Shift(-0.4),
        )
        self.body = Residual(res_blocks + [Conv2dInit(maps, maps, (3, 3), padding=1)], scale=None)
        self.tail = nn.Sequential(
            Conv2dInit(maps, factor ** 2 * maps, (3, 3), padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(negative_slope=LRELU_SLOPE),
            Conv2dInit(maps, 1, (3, 3), padding=1),
            Shift(0.4),
        )

    def forward(self, x):
        return self.tail(self.body(self.head(x)))

    def modules(self):
        return [self.body.body[i] for i in range(len(self.body.body))]

    def depth(self):
        return len(self.body.body) - 1

    def add_layers(self):
        modules = self.modules()
        modules = modules[:-1] + [Residual(Block(self.maps)) for _ in range(2)] + modules[-1:]
        self.body.body = nn.Sequential(*modules)
        if modules[-1].weight.data.is_cuda:
            self.body = self.body.cuda()
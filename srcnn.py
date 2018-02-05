import torch
import torch.nn as nn
import numpy as np

CONV_NORMAL_STD = 0.001
def Conv2dNormal(*args, **kwargs):
    result = nn.Conv2d(*args, **kwargs)
    result.weight.data.normal_(mean=0, std=CONV_NORMAL_STD)
    return result

class CTSRCNN(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        self.stack = nn.Sequential(
            Conv2dNormal(1, 64, (9, 9), padding=4), self.lrelu,
            Conv2dNormal(64, 32, (5, 5), padding=2), self.lrelu,
            Conv2dNormal(32, 1, (5, 5), padding=2),
        )

    def forward(self, inp):
        return self.stack(inp)

    def add_layers(self):
        modules = [self.stack[i] for i in range(len(self.stack))]
        modules = modules[:-1] + [
            Conv2dNormal(32, 32, (3, 3), padding=1), self.lrelu,   
            Conv2dNormal(32, 32, (3, 3), padding=1), self.lrelu,
        ] + modules[-1:]
        self.stack = nn.Sequential(*modules)

    def depth(self):
        return len(self.stack) // 2 + 1


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
            Conv2dNormal(1, 64, (9, 9), padding=4), nn.PReLU(num_parameters=64, init=0.1),
            Conv2dNormal(64, 32, (5, 5), padding=2), nn.PReLU(num_parameters=32, init=0.1),
            Conv2dNormal(32, 1, (5, 5), padding=2),
        )

    def forward(self, inp):
        return self.stack(inp)

    def modules(self):
        return [self.stack[i] for i in range(len(self.stack))]

    def add_layers(self):
        modules = self.modules()
        modules = modules[:-1] + [
            Conv2dNormal(32, 32, (3, 3), padding=1), nn.PReLU(num_parameters=32, init=0.1),   
            Conv2dNormal(32, 32, (3, 3), padding=1), nn.PReLU(num_parameters=32, init=0.1),
        ] + modules[-1:]
        self.stack = nn.Sequential(*modules)
        if modules[-1].weight.data.is_cuda:
            self.stack = self.stack.cuda()

    # Count (from back) how many layers have been trimmed.
    # 0 if untrimmed; 2 if double-trimmed once.
    def trim_count(self):
        modules = self.modules()
        for i, module in enumerate(modules[-3::-2]):
            assert isinstance(module, nn.Conv2d)
            if module.in_channels == 1 and module.out_channels == 64:
                return i
            elif module.out_channels == 32:
                return i
        return self.depth()

    # trim layer's outgoing connection
    def trim_layer(self, i):
        modules = self.modules()
        layer_idx = 2 * i if i >= 0 else 2 * i - 1  # 2 -> (4, 6); -1 -> (-3, -5)
        current = modules[layer_idx]
        prelu = modules[layer_idx + 1]
        next = modules[layer_idx + 2]

        assert isinstance(current, nn.Conv2d)
        assert isinstance(prelu, nn.PReLU)
        assert isinstance(next, nn.Conv2d)
        assert current.out_channels == next.in_channels
        channels = current.out_channels
        selected = torch.randperm(channels)[:channels // 2]
        new_current = nn.Conv2d(current.in_channels, channels // 2,
                                kernel_size=current.kernel_size,
                                stride=current.stride,
                                padding=current.padding,
                                dilation=current.dilation,
                                bias=current.bias is not None)
        new_prelu = nn.PReLU(num_parameters=1 if prelu.num_parameters == 1 else prelu.num_parameters // 2)
        new_next = nn.Conv2d(channels // 2, next.out_channels,
                             kernel_size=next.kernel_size,
                             stride=next.stride,
                             padding=next.padding,
                             dilation=next.dilation,
                             bias=next.bias is not None)

        if current.weight.data.is_cuda:
            new_current, new_prelu, new_next = new_current.cuda(), new_prelu.cuda(), new_next.cuda()
            selected = selected.cuda()

        torch.index_select(current.weight.data, 0, selected, out=new_current.weight.data)
        if current.bias is not None:
            torch.index_select(current.bias.data, 0, selected, out=new_current.bias.data)
        torch.index_select(prelu.weight.data, 0, selected, out=new_prelu.weight.data)
        torch.index_select(next.weight.data, 1, selected, out=new_next.weight.data)

        modules[layer_idx] = new_current
        modules[layer_idx + 1] = new_prelu
        modules[layer_idx + 2] = new_next
        self.stack = nn.Sequential(*modules)

    def trim(self):
        trim_count = self.trim_count()
        self.trim_layer(-trim_count - 1)
        self.trim_layer(-trim_count - 2)

    def depth(self):
        return len(self.stack) // 2 + 1

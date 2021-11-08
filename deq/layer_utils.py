'''
https://github.com/locuslab/deq/blob/master/lib/layer_utils.py
'''
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn


def list2vec(z1_list):
    """Convert list of tensors to a flattened vector tensor"""
    a = torch.cat([elem.reshape(1, -1, 1) for elem in z1_list], dim=1)
    # print(a.shape)
    return a


def vec2list(z1, shapes):
    """Convert a vector back to a list, via the cutoffs specified"""
    z1_list = []
    idx = 0
    # print(shapes)
    for i in range(len(shapes)):
        n = np.prod(shapes[i])
        z1_list.append(z1[:, idx:idx+n, :].view(shapes[i]))
        idx += n
    return z1_list


def conv3x3(in_planes, out_planes, stride=1, bias=False, **kwargs):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias, **kwargs)

def conv5x5(in_planes, out_planes, stride=1, bias=False, **kwargs):
    """5x5 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride, padding=2, bias=bias, **kwargs)


def norm_diff(new, old, show_list=False):
    if show_list:
        return [(new[i] - old[i]).norm().item() for i in range(len(new))]
    return np.sqrt(sum((new[i] - old[i]).norm().item()**2 for i in range(len(new))))
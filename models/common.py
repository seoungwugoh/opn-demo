from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models
 
# general libs
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import time
import tqdm
import os
import argparse
import copy
import sys

sys.path.insert(0, '../utils/')
from utils.helpers import *

##########################################
############   Generic   #################
##########################################

def pad_divide_by(in_list, d, in_size):
    out_list = []
    h, w = in_size
    if h % d > 0:
        new_h = h + d - h % d
    else:
        new_h = h
    if w % d > 0:
        new_w = w + d - w % d
    else:
        new_w = w
    lh, uh = int((new_h-h) / 2), int(new_h-h) - int((new_h-h) / 2)
    lw, uw = int((new_w-w) / 2), int(new_w-w) - int((new_w-w) / 2)
    pad_array = (int(lw), int(uw), int(lh), int(uh))
    for inp in in_list:
        out_list.append(F.pad(inp, pad_array))
    return out_list, pad_array


class ConvGRU(nn.Module):
    def __init__(self, mdim, kernel_size=3, padding=1):
        super(ConvGRU, self).__init__()
        self.convIH = nn.Conv2d(mdim, 3*mdim, kernel_size=kernel_size, padding=padding)
        self.convHH = nn.Conv2d(mdim, 3*mdim, kernel_size=kernel_size, padding=padding)

    def forward(self, input, hidden_tm1):
        if hidden_tm1 is None:
            hidden_tm1 = torch.zeros_like(input)
        gi = self.convIH(input)
        gh = self.convHH(hidden_tm1)
        i_r, i_i, i_n = torch.chunk(gi, 3, dim=1)
        h_r, h_i, h_n = torch.chunk(gh, 3, dim=1)
        resetgate = F.sigmoid(i_r + h_r) # reset
        inputgate = F.sigmoid(i_i + h_i) # update
        newgate = F.tanh(i_n + resetgate * h_n)
        # hidden_t = inputgate * hidden_tm1 + (1-inputgate)*newgate
        hidden_t = newgate + inputgate * (hidden_tm1 - newgate)
        return hidden_t


def F_upsample3d(x, size=None, scale_factor=None, mode='nearest', align_corners=None):
    num_frames = x.size()[2]
    up_s = []
    for f in range(num_frames):
        up = F.upsample(x[:,:,f], size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners)
        up_s.append(up)
    ups = torch.stack(up_s, dim=2) 
    return ups

def F_upsample(x, size=None, scale_factor=None, mode='nearest', align_corners=None):
    if x.dim() == 5: # 3d
        return F_upsample3d(x, size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners)
    else: 
        return F.upsample(x, size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners)


class GatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, activation=None):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups, bias)
        self.gating_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                   stride, padding, dilation, groups, bias)
        init_He(self)
        self.activation = activation

    def forward(self, input):
        # O = act(Feature) * sig(Gating)
        feature = self.input_conv(input)
        if self.activation:
            feature = self.activation(feature)
        gating = F.sigmoid(self.gating_conv(input))
        return feature * gating

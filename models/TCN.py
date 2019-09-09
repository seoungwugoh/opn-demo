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
import random
 
sys.path.insert(0, '.')
from .common import *
sys.path.insert(0, '../utils/')
from utils.helpers import *
 
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = GatedConv2d(8, 32, kernel_size=5, stride=1, padding=2, activation=nn.LeakyReLU(negative_slope=0.2)) # 2
        self.conv21 = GatedConv2d(32, 64, kernel_size=3, stride=2, padding=1, activation=nn.LeakyReLU(negative_slope=0.2)) # 2
        self.conv2 = GatedConv2d(64, 64, kernel_size=3, stride=1, padding=1, activation=nn.LeakyReLU(negative_slope=0.2)) # 2
        self.conv23 = GatedConv2d(64, 128, kernel_size=3, stride=2, padding=1, activation=nn.LeakyReLU(negative_slope=0.2)) # 4
        self.conv3a = GatedConv2d(128, 128, kernel_size=3, stride=1, padding=1, activation=nn.LeakyReLU(negative_slope=0.2)) # 4
        self.conv3b = GatedConv2d(128, 128, kernel_size=3, stride=1, padding=1, activation=nn.LeakyReLU(negative_slope=0.2)) # 4
        self.conv34 = GatedConv2d(128, 256, kernel_size=3, stride=2, padding=1, activation=nn.LeakyReLU(negative_slope=0.2)) # 4
        self.conv4a = GatedConv2d(256, 256, kernel_size=3, stride=1, padding=1, activation=nn.LeakyReLU(negative_slope=0.2)) # 4
        self.conv4b = GatedConv2d(256, 256, kernel_size=3, stride=1, padding=1, activation=nn.LeakyReLU(negative_slope=0.2)) # 4
        
        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, O_tm1, H_tm1, F_t, H_t):
        o = (O_tm1 - self.mean) / self.std
        f = (F_t - self.mean) / self.std
        x = torch.cat([o, H_tm1, f, H_t], dim=1)
        
        s1 = self.conv1(x)
        x = self.conv21(s1)
        s2 = self.conv2(x)
        x = self.conv23(s2)
        x = self.conv3a(x)
        s3 = self.conv3b(x)
        x = self.conv34(s3)
        x = self.conv4a(x)
        x = self.conv4b(x)

        return x, s3, s2, s1


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv4b = GatedConv2d(256, 256, kernel_size=3, stride=1, padding=2, dilation=2, activation=nn.LeakyReLU(negative_slope=0.2)) # 4
        self.conv4a = GatedConv2d(256, 256, kernel_size=3, stride=1, padding=1, activation=nn.LeakyReLU(negative_slope=0.2)) # 4
        self.conv43 = GatedConv2d(256, 128, kernel_size=3, stride=1, padding=1, activation=nn.LeakyReLU(negative_slope=0.2)) # 2
        self.conv3b = GatedConv2d(128, 128, kernel_size=3, stride=1, padding=2, dilation=2, activation=nn.LeakyReLU(negative_slope=0.2)) # 4
        self.conv3a = GatedConv2d(128, 128, kernel_size=3, stride=1, padding=1, activation=nn.LeakyReLU(negative_slope=0.2)) # 4
        self.conv32 = GatedConv2d(128, 64, kernel_size=3, stride=1, padding=1, activation=nn.LeakyReLU(negative_slope=0.2)) # 2
        self.conv2 = GatedConv2d(64, 64, kernel_size=3, stride=1, padding=1, activation=nn.LeakyReLU(negative_slope=0.2)) # 2
        self.conv21 = GatedConv2d(64, 32, kernel_size=3, stride=1, padding=1, activation=nn.LeakyReLU(negative_slope=0.2)) # 1
        self.conv1 = GatedConv2d(32, 3, kernel_size=5, stride=1, padding=2, activation=None) # 1

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, hid, s3, s2, s1):
        x = self.conv4b(hid)
        x = self.conv4a(x)
        x = F.upsample(x, scale_factor=2, mode='nearest') # 2
        x = self.conv43(x)
        x = self.conv3b(x+s3)
        x = self.conv3a(x)
        x = F.upsample(x, scale_factor=2, mode='nearest') # 2
        x = self.conv32(x)
        x = self.conv2(x+s2)
        x = F.upsample(x, scale_factor=2, mode='nearest') # 2
        x = self.conv21(x)
        # x = self.conv1(x) # t4
        x = self.conv1(x+s1) # t3
        p = (x * self.std) + self.mean
        return p

class TCN(nn.Module):
    def __init__(self):
        super(TCN, self).__init__()
        self.Encoder = Encoder() 
        self.GRU = ConvGRU(256, kernel_size=3, padding=1)
        self.Decoder = Decoder()

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('mean3d', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1,1))


    def forward(self, prev_frame, prev_hole, curr_frame, curr_hole, prev_hidden): 
        # padding
        (prev_frame, prev_hole, curr_frame, curr_hole), pad = pad_divide_by([prev_frame, prev_hole, curr_frame, curr_hole], 8, (curr_frame.size()[2], curr_frame.size()[3]))
        batch_size, _, height, width = prev_frame.size()

        # encoding and decoding...
        enc, s3, s2, s1 = self.Encoder(prev_frame, prev_hole, curr_frame, curr_hole)
        # read.
        hidden = self.GRU(enc, prev_hidden)
        # decode
        pped = self.Decoder(hidden, s3, s2, s1)

        if pad[2]+pad[3] > 0:
            pped = pped[:,:,pad[2]:-pad[3],:]
        if pad[0]+pad[1] > 0:
            pped = pped[:,:,:,pad[0]:-pad[1]]
            
        pped = torch.clamp(pped, 0, 1)
        return pped, hidden
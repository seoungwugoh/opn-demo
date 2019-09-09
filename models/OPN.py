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
        self.conv12 = GatedConv2d(5, 64, kernel_size=5, stride=2, padding=2, activation=nn.LeakyReLU(negative_slope=0.2)) # 2
        self.conv2 = GatedConv2d(64, 64, kernel_size=3, stride=1, padding=1, activation=nn.LeakyReLU(negative_slope=0.2)) # 2
        self.conv23 = GatedConv2d(64, 128, kernel_size=3, stride=2, padding=1, activation=nn.LeakyReLU(negative_slope=0.2)) # 4
        self.conv3a = GatedConv2d(128, 128, kernel_size=3, stride=1, padding=1, activation=nn.LeakyReLU(negative_slope=0.2)) # 4
        self.conv3b = GatedConv2d(128, 128, kernel_size=3, stride=1, padding=2, dilation=2, activation=nn.LeakyReLU(negative_slope=0.2)) # 4
        self.conv3c = GatedConv2d(128, 128, kernel_size=3, stride=1, padding=4, dilation=4, activation=nn.LeakyReLU(negative_slope=0.2)) # 4
        self.conv3d = GatedConv2d(128, 128, kernel_size=3, stride=1, padding=8, dilation=8, activation=nn.LeakyReLU(negative_slope=0.2)) # 4
        self.key3 = GatedConv2d(128, 128, kernel_size=3, stride=1, padding=1, activation=None) # 4
        self.val3 = GatedConv2d(128, 128, kernel_size=3, stride=1, padding=1, activation=None) # 4
        
        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, in_f, in_v, in_h):
        f = (in_f - self.mean) / self.std
        x = torch.cat([f, in_v, in_h], dim=1)
        x = self.conv12(x)
        x = self.conv2(x)
        x = self.conv23(x)
        x = self.conv3a(x)
        x = self.conv3b(x)
        x = self.conv3c(x)
        x = self.conv3d(x)
        k = self.key3(x)
        v = self.val3(x)
        return k, v

class MaskedRead(nn.Module):
    def __init__(self):
        super(MaskedRead, self).__init__()

    def forward(self, qkey, qval, qmask, mkey, mval, mmask):
        '''
        read for *mask area* of query from *mask area* of memory
        '''

        B, Dk, _, H, W = mkey.size()
        _, Dv, _, _, _ = mval.size()
        # key: b,dk,t,h,w
        # value: b,dv,t,h,w
        # mask: b,1,t,h,w
        for b in range(B):
            # exceptions
            if qmask[b,0].sum() == 0 or mmask[b,0].sum() == 0: 
                # print('skipping read', qmask[b,0].sum(), mmask[b,0].sum())
                # no query or mask pixels -> skip read
                continue
            qk_b = qkey[b,:,qmask[b,0]] # dk, Nq
            mk_b = mkey[b,:,mmask[b,0]] # dk, Nm
            mv_b = mval[b,:,mmask[b,0]] # dv, Nm 
            # print(mv_b.shape)

            p = torch.mm(torch.transpose(mk_b, 0, 1), qk_b) # Nm, Nq
            p = p / math.sqrt(Dk)
            p = F.softmax(p, dim=0)

            read = torch.mm(mv_b, p) # dv, Nq
            # qval[b,:,qmask[b,0]] = read # dv, Nq
            qval[b,:,qmask[b,0]] = qval[b,:,qmask[b,0]] + read # dv, Nq
            
        return qval


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv3d = GatedConv2d(128, 128, kernel_size=3, stride=1, padding=8, dilation=8, activation=nn.LeakyReLU(negative_slope=0.2)) # 4
        self.conv3c = GatedConv2d(128, 128, kernel_size=3, stride=1, padding=4, dilation=4, activation=nn.LeakyReLU(negative_slope=0.2)) # 4
        self.conv3b = GatedConv2d(128, 128, kernel_size=3, stride=1, padding=2, dilation=2, activation=nn.LeakyReLU(negative_slope=0.2)) # 4
        self.conv3a = GatedConv2d(128, 128, kernel_size=3, stride=1, padding=1, activation=nn.LeakyReLU(negative_slope=0.2)) # 4
        self.conv32 = GatedConv2d(128, 64, kernel_size=3, stride=1, padding=1, activation=nn.LeakyReLU(negative_slope=0.2)) # 2
        self.conv2 = GatedConv2d(64, 64, kernel_size=3, stride=1, padding=1, activation=nn.LeakyReLU(negative_slope=0.2)) # 2
        self.conv21 = GatedConv2d(64, 3, kernel_size=5, stride=1, padding=2, activation=None) # 1

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, x):
        x = self.conv3d(x)
        x = self.conv3c(x)
        x = self.conv3b(x)
        x = self.conv3a(x)
        x = F.upsample(x, scale_factor=2, mode='nearest') # 2
        x = self.conv32(x)
        x = self.conv2(x)
        x = F.upsample(x, scale_factor=2, mode='nearest') # 2
        x = self.conv21(x)
        p = (x * self.std) + self.mean
        return p

class OPN(nn.Module):
    def __init__(self, mode='Train', CPU_memory=False, thickness=8):
        super(OPN, self).__init__()
        self.Encoder = Encoder() 
        self.MaskedRead = MaskedRead()
        self.Decoder = Decoder()

        self.thickness = thickness

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('mean3d', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1,1))


    def memorize(self, frames, valids, dists):
        '''
        encode every frame of *valid* area into key:value
        Done once as initialization
        '''
        
         # padding
        (frames, valids, dists), pad = pad_divide_by([frames, valids, dists], 4, (frames.size()[3], frames.size()[4]))
        
        # make hole
        holes = (dists > 0).float()
        frames = (1-holes)*frames + holes*self.mean3d
        batch_size, _, num_frames, height, width = frames.size()

        # encoding...
        key_ = []
        val_ = []
        for t in range(num_frames):
            key, val = self.Encoder(frames[:,:,t], valids[:,:,t], holes[:,:,t])
            key_.append(key)
            val_.append(val)

        keys = torch.stack(key_, dim=2)
        vals = torch.stack(val_, dim=2)

        hols = (F_upsample3d(holes, size=(int(height/4), int(width/4)), mode='bilinear', align_corners=False) > 0)
        return keys, vals, hols

    def read(self, mkey, mval, mhol, frame, valid, dist): 
        ''' 
        ## assume single frame query
        1) encode current status of frames -> query
        2) read from memmories (computed calling 'memorize')
        3) decode readed feature
        4) compute loss on peel area
        '''
        thickness = self.thickness

        # padding
        (frame, valid, dist), pad = pad_divide_by([frame, valid, dist], 4, (frame.size()[2], frame.size()[3]))
        batch_size, _, height, width = frame.size()
        # make hole and peel..
        hole = (dist > 0).float()
        peel = hole * (dist <= thickness).float()
        next_dist =  torch.clamp(dist-thickness, 0, 9999)
        # get 1/4 scale mask
        peel3 = (F.upsample(peel, size=(int(height/4), int(width/4)), mode='bilinear', align_corners=False) >= 0.5)

        frame = (1-hole)*frame + hole*self.mean

        # reading and decoding...
        qkey, qval = self.Encoder(frame, valid, hole)
        qpel = peel3
        # read.
        read = self.MaskedRead(qkey, qval, qpel, mkey, mval, 1-mhol)
        # decode
        pred = self.Decoder(read)
        comp = (1-peel)*frame + peel*pred # fill peel area
        
        if pad[2]+pad[3] > 0:
            comp = comp[:,:,pad[2]:-pad[3],:]
            next_dist = next_dist[:,:,pad[2]:-pad[3],:]
        if pad[0]+pad[1] > 0:
            comp = comp[:,:,:,pad[0]:-pad[1]]
            next_dist = next_dist[:,:,:,pad[0]:-pad[1]]
            
        comp = torch.clamp(comp, 0, 1)
        return comp, next_dist



    def forward(self, *args, **kwargs):
        # print(len(args), len(kwargs))
        if len(args) == 3:
            return self.memorize(*args)
        else:
            return self.read(*args, **kwargs)
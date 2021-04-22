# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 17:11:37 2021

@author: Kim
"""

import torch
import torch.nn as nn


class VGG(nn.Module):
    def __init__(self, config):
        super(VGG, self).__init__()

        self.in_channel = config[0]
        self.out_channel = config[1]
        self.kernel_size = config[2]
        self.layer_num = config[3]
        self.dropout_rate = config[4]
        self.class_num = config[5]
        self.pool_size = config[6]
        
        self.make_layer = self._make_layers()
        self.dropout = nn.Dropout2d(self.dropout_rate)
        
        self.linear = nn.Linear(256, self.class_num)
        self.relu = nn.ReLU()
        
        
    def forward(self, x):
        x = self.make_layer(x)
        x = x.view(x.size(0),-1)
        
        param_num = x.shape[1]
        linear_1 = nn.Linear(param_num, 256)

        x = self.relu(linear_1(x))
        x = self.dropout(x)
        x = self.linear(x)
        return x
    
        
    def _make_layers(self):
        layers = []
        in_channels = self.in_channel
        out_channels = self.out_channel
        
        for i in range(self.layer_num):
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=self.kernel_size, padding=1),
                       nn.BatchNorm2d(out_channels),
                       nn.ReLU(out_channels),
                       nn.MaxPool2d(self.pool_size, self.pool_size)]
            in_channels = out_channels
            out_channels = in_channels * 2
        return nn.Sequential(*layers)


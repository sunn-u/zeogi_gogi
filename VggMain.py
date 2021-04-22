# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 17:13:05 2021

@author: Kim
"""

import torch
from torchsummary import summary
import VggModel

class DataLoaderMain():
    def __init__(self):
        self.config = [1, 32, 3, 1, 0.5, 10, 2]
        self.image_shape = (1,200,200)
        self.vgg = VggModel.VGG(self.config)
        
    def main(self):
        model = self.vgg
        
        print(summary(model, self.image_shape))
        
        
if __name__ == '__main__':
    main_class = DataLoaderMain()
    main_class.main()
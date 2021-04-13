# -*- coding: utf-8 -*-
"""
created by sue
"""


import torch
from torchvision import transforms
from torch.utils.data.dataset import random_split
import random
import numpy as np
import CustomDataset

SEED = 0 
random.seed(SEED) 
np.random.seed(SEED) 
torch.manual_seed(SEED)


class DataLoader():
    import CustomDataset

    def __init__(self, config):
        self.path = config['path']
        self.weight = int(config['weight'])
        self.height = int(config['height'])
        self.beatch_size = int(config['beatch_size'])
        
    def Data_Loader(self):
        data_transforms = transforms.Compose([transforms.Resize((self.weight, self.height)),
                          transforms.ToTensor()])
        data_class = CustomDataset.CustomDataset(self.path, transforms=data_transforms)
        
        return data_class
    
    def DataSplit(self, data_class):
        train_len = int(0.8*len(data_class))
        valid_len = len(data_class) - train_len
        train_dataset, val_dataset = random_split(data_class, [train_len,valid_len])
        
        train_set = torch.utils.data.DataLoader(train_dataset, batch_size=self.beatch_size, shuffle=True)
        val_set = torch.utils.data.DataLoader(val_dataset, batch_size=self.beatch_size, shuffle=False)
        
        return train_set, val_set
        
        
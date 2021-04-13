# -*- coding: utf-8 -*-
"""
created by sue
"""

import DataLoader
import configparser


class DataLoaderMain():
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.read('C:/Users/Kim/config_data.ini', encoding='UTF-8')
        self.dataloader = DataLoader.DataLoader(self.config['data_loader'])
        
    def main(self):
        data_class = self.dataloader.Data_Loader()        
        train_set, val_set = self.dataloader.DataSplit(data_class)
        
        print('train_set: ', len(train_set))
        print('validation_set: ', len(val_set))
        
        
if __name__ == '__main__':
    main_class = DataLoaderMain()
    main_class.main()
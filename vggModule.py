import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
from configs import Configs


class Vgg(nn.Module):

    def __init__(self, conv, num_classes=100, init_weights=True):
        super(Vgg, self).__init__()  # pytorch에서 class 형태의 모델은 항상 nn.Module을 상속받아야 하며, 
                                     # super(모델명, self).init()을 통해 nn.Module.init()을 실행시키는 코드가 필요

        self.features = conv    # features
        self.avgpool = nn.AvgPool2d(7)
        self.classifier = nn.Linear(512, 7)   # 7은 num_classes
        '''
        self.classifier = nn.Sequential(
          nn.Linear(512*7*7, 4096),
          nn.ReLU(True),
          nn.Dropout(),
          nn.Linear(4096, 4096),
          nn.ReLU(True),
          nn.Dropout(),
          nn.Linear(4096, num_classes),
        )
        '''

        if init_weights:
            self._initialize_weights()
  
    def forward(self, x):  # 모델이 학습 데이터를 입력 받아서 forward prop을 진행시키는 함수
        #features = self.conv(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)   # x = torch.flatten(x, 1) 이걸로도 가능한것인지
        x = self.classifier(x)
        #x = self.softmax(x)
        return x

    def _initialize_weights(self):  # 가중치 초기화 
        for m in self.modules():  # self.modules()는 모델 클래스에서 정의된 layer들을 iterable로 차례로 반환
            if isinstance(m, nn.Conv2d):  # isinstance()는 차례로 layer를 입력하여 layer의 형태를 반환(nn.cnv2d, nn,BatchNorm2d 등..)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')    # torch.nn.init.kaiming_normal_ 는  He initialization 을 제공하는 함수 
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)  # torch.nn.init.normal_(tensor, mean=0.0, std=1.0)은 tensor을 mean, std의 normal distribution으로 초기화
                nn.init.constant_(m.bias, 0)  # torch.nn.init.constant_(tensor, val)은 tensor을 val로 초기화



def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 1

    for v in cfg:
        if v == 'M':  # max pooling
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
  
    return nn.Sequential(*layers)

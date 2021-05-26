# Initial
import torch
import torchvision
import torchvision.transforms as transforms

## torch library init
from skimage import feature
from torchvision.models.vgg import make_layers

torch.cuda.is_available()
# gpu_available = torch.cuda.is_available()
# print(gpu_available)

# dataloader

import torchvision.transforms as transforms

import torch.nn as nn

'''
transforms = transforms.Compose([
    transforms.Resize(224),  # vgg에서 이미지 대상 크기 224*224
    transforms.ToTensor(),  # 데이터를 pytorch에서 사용하기 위한 Tensor자료 구조로 변환
    transforms.Normalize((0.5,),(0,5,)), # 데이터 normalizing (특정부분 튀는것 막아줌)
])
'''

### custom dataloader
'''
trainset = torchvision.datasets.STL10(root='./data', split='train', download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.STL10(root='./data', split='test', download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)


'''

# vgg 모델 구현
class Vgg(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(Vgg, self).__init__()
        self.features = features
        #self.avgpool = nn.AdaptiveAvgPool2d(7)
        self.avgpool = nn.AvgPool2d(7)
        # self.classifier = nn.Sequential( nn.Linear(512*7*7, 4096), nn.ReLU(True),
        #                                  nn.Dropout(),
        #                                  nn.Linear(4096, 4096),
        #                                  nn.ReLU(True),
        #                                  nn.Dropout(),
        #                                  nn.Linear(4096, num_classes), )
        self.classifier = nn.Linear(512, num_classes)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)




def Make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 1

    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)





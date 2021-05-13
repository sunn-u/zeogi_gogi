from torchvision.utils import save_image
from MyDataLoader import *
from torchvision.models.vgg import make_layers


import matplotlib.pyplot as plt
import numpy as np

from vggModule import VGG
from configs import Configs
from train import train


def main():
    trainloader, testloader = make_train_val_set(Configs.data_root)
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    cfg = {  # 8 + 3 =11 == vgg11
	    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	    # 10 + 3 = vgg 13
	    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	    # 13 + 3 = vgg 16
	    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
	    # 16 +3 =vgg 19
	    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
	}
    conv = make_layers(cfg[Configs.model_config], batch_norm=True)
    model = Vgg(conv, num_classes=Configs.class_num, init_weights=True)
    print(model)

    train(model, trainloader, testloader)



if __name__ == '__main__':
	
    main()

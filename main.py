from torchvision.utils import save_image
from Data_Loader import train_loader
from MyDataLoader import make_train_val_set
from torchvision.models.vgg import make_layers

import splitdata
from dataloader import *
import matplotlib.pyplot as plt
import numpy as np

from vggModule import VGG


def main():
	make_train_val_set(opt.data)
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

	conv = make_layers(cfg['D'], batch_norm=True)
	model = VGG(conv, num_classes=7, init_weights=True)
	model


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data', type=str, default='/content/gdrive/Shareddrives/zeogi_gogi/dataset/', help='dataset path')
	opt = parser.parse_args()
	
	main()
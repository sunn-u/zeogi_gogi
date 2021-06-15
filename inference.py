import torch.optim as optim
import time
import cv2
from PIL import Image
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
#import torchvision.transforms as transforms
from skimage import io, transform
from torchvision import transforms

import torch.nn as nn
import torch.nn.functional as F
from configs import Configs
from vggModule import *
from MyDataLoader import *

classes = ['fear', 'happy', 'sad', 'neutral', 'disgust', 'surprise', 'angry']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
######## dropout을 하지않은 flatten을 바로 한 모델은 shape 에러 발생 (try except를 사용하던지 해서 모든 모델에 가능하도록 해야함)#############
#model = torch.load("/data/FoodDetection/Object_Detection/yolov5-test/ssd/models/ep100/vgg16_epoch99_batch393_accuracy92.875.pt")
######## dropout을 한 모델은 결과 도출 ##########
model = torch.load("/data/FoodDetection/Object_Detection/yolov5-test/ssd/models/ep100_vgg16_dropout0.7/vgg11_epoch85_accuracy99.952.pt")

def test(path):
    
    aug_f = transform = transforms.Compose(
                [transforms.ToPILImage(),
                transforms.Resize((224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))])

    images = cv2.imread(path)
    #images = cv2.imread(Configs.test_image_path)
    #images = torch.tensor(images)
    X_test = aug_f(images)
    X_test = torch.tensor(X_test).unsqueeze(dim=1)
    X_single_data = X_test.float().to(device)
    #X_single_data = images.view(-1, 224 * 224).float().to(device)

    single_prediction = model(X_single_data)
    print('Prediction: ', torch.argmax(single_prediction).item(), classes[torch.argmax(single_prediction).item()])
    #print('Prediction: ', torch.argmax(single_prediction, 1).item())
    #print('Prediction: ', torch.argmax(single_prediction, 1).item(), classes[torch.argmax(single_prediction, 1).item()])

    #plt.imshow(images.view(224, 224), cmap='Greys', interpolation='nearest')
    #plt.imshow(images.view(224, 224))
    plt.imshow(images)
    img = plt.show()
    
    return torch.argmax(single_prediction).item(), classes[torch.argmax(single_prediction).item()]


test(Configs.test_image_path)
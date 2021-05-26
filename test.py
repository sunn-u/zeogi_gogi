# 테스트 데이터를 사용하여 모델을 테스트한다.
import torch.optim as optim
import time
import random
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
from configs import Configs
from vggModule import *
from MyDataLoader import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
trainloader, testloader = make_train_val_set(Configs.data_root)
dataiter = iter(testloader)
images, labels = dataiter.next()

with torch.no_grad(): # torch.no_grad()를 하면 gradient 계산을 수행하지 않는다.
    X_test = images.float().to(device)
    #X_test = images.view(-1, 224 * 224).float().to(device)
    #print(X_test.shape)
    Y_test = labels.to(device)
    #print(Y_test.shape)

    model = torch.load("/data/FoodDetection/Object_Detection/yolov5-test/ssd/models/ep10/vgg16_epoch10_batch393_accuracy73.223.pt")
    model.eval()
    prediction = model(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())

    r = random.randint(0, len(testloader) - 1)
    X_single_data = images[r:r + 1].float().to(device)
    #X_single_data = images[r:r + 1].view(-1, 224 * 224).float().to(device)
    Y_single_data = labels[r:r + 1].to(device)

    print('Label: ', Y_single_data.item())
    single_prediction = model(X_single_data)
    print('Prediction: ', torch.argmax(single_prediction, 1).item())

    plt.imshow(images[r:r + 1].view(224, 224), cmap='Greys', interpolation='nearest')
    plt.show()


import torch.optim as optim
import time
import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
from configs import Configs

def train():

    model.train()

    classes = ('fear', 'happy', 'sad', 'neutral', 'disgust', 'surprise', 'angry')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=0.0001)
    #device = torch.device("cuda:0")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #cuda = device.type != 'cpu'

    start_time = time.time()
    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        #dataiter = iter(trainloader)
        #images, labels = dataiter.next()
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()


            # forward + backward + optimize
            model = model.float()
            model = model.cuda()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if(loss.item() > 1000):
                print(loss.item())
                for param in model.parameters():
                    print(param.data)
            # print statistics
            running_loss += loss.item()
            if i % 50 == 49:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 50))
                running_loss = 0.0

    torch.save(model, '/content/drive/MyDrive/vgg16.pt')
    torch.save(model.state_dict(), '/content/drive/MyDrive/vgg16.pth')
    print(time.time()-start_time)
    print('Finished Training')

    
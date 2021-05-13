import torch.optim as optim
import time
import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
from configs import Configs
from vggModule import *
from MyDataLoader import *

def train(model, trainloader, testloader):

    model.train()

    classes = ('fear', 'happy', 'sad', 'neutral', 'disgust', 'surprise', 'angry')
    import torch.optim as optim
    import time

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=0.00001)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    start_time = time.time()
    for epoch in range(2):  # loop over the dataset multiple times
        trainLoss = 0.0
        trainSize = 0
        trainCorrect = 0
        
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            model = model.float()
            #outputs,f = model(inputs)
            model = model.cuda()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            trainLoss += loss.item()

            predicted = outputs.max(1)[1]
            #print(predicted)
            #print(labels)
      
            trainSize += labels.shape[0]
            trainCorrect += predicted.eq(labels.view_as(predicted)).sum().item()
            trainAccuracy = (trainCorrect / trainSize) * 100

        torch.save(model, '/content/drive/MyDrive/vgg16_epoch{}_batch{}_accuracy{:.3f}.pt'.format(epoch+1, i+1, trainAccuracy))
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': trainLoss,
            }, '/content/drive/MyDrive/vgg16_epoch{}_batch{}_accuracy{:.3f}.pth'.format(epoch+1, i+1, trainAccuracy))

        #print('epoch {} batch {} train_loss {}  accuracy {}'.format(epoch+1, i+1, trainLoss / trainSize, trainAccuracy))
        print('Epoch {}/{}'.format(epoch, epoch_nb))
        print("---------")
        print('train Loss: {}  Acc: {}'.format(trainLoss / trainSize, trainAccuracy))
        trainLoss = 0.0 


        #if i % 5 == 0:
        with torch.no_grad():
            valLoss = 0
            valSize = 0
            valCorrect = 0

            for j, data in enumerate(testloader, 0):
                # get the inputs
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                # forward + backward + optimize
                model = model.float()
                #outputs,f = model(inputs)
                model = model.cuda()
                outputs = model(inputs)

                valLoss += criterion(outputs, labels).item()

                predicted = outputs.max(1)[1]
                valSize += labels.shape[0]
                valCorrect += predicted.eq(labels.view_as(predicted)).sum().item()
                valAccuracy = (valCorrect / valSize) * 100
            
            #print('validation_loss {} validation_accuracy {}'.format(valLoss / valSize, valAccuracy))
            print('val Loss: {}  Acc: {}'.format(valLoss / valSize, valAccuracy))
            valLoss = 0.0



    #torch.save(model, '/content/drive/MyDrive/vgg16.pt')
    #torch.save(model.state_dict(), '/content/drive/MyDrive/vgg16.pth')
    print(time.time()-start_time)
    print('Finished Training')


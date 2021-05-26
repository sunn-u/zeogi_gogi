# from torchvision.models.vgg import make_layers
import splitdata
from dataloader import *
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
# from vggModule import VGG, MakeLayer
from vggModule import Vgg, Make_layers
import torch.nn as nn
import torch
import sys
import time





def dataloader():
    # splitdata.split_dataset_into_3('/Users/ahrim/Desktop/dataset', 0.6, 0.2)

    classes = ('angry','disgust','fear','happy','neutral','sad','surprise')

    trainloader = torch.utils.data.DataLoader(
        MyFaceSet(
            DATA_PATH_TRAINING_LIST,
            classes,
            transform=transform
        ),
        batch_size=4,
        shuffle=True
    )

    testloader = torch.utils.data.DataLoader(
        MyFaceSet(
            DATA_PATH_TESTING_LIST,
            classes,
            transform=transform
        ),
        batch_size=4,
        shuffle=False
    )

    valloader = torch.utils.data.DataLoader(
        MyFaceSet(
            DATA_PATH_VALIDATION_LIST,
            classes,
            transform=transform
        ),
        batch_size=4,
        shuffle=False
    )
    return trainloader, testloader, valloader



# class Main():
#     def __init__(self):
#         self.makelayer = MakeLayer()

def main():
    trainData, testData, valData = dataloader()
    dataiter = iter(trainData)
    images, labels = dataiter.next()

    cfg = {
        #8 + 3 =11 == vgg11
        'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        # 10 + 3 = vgg 13
        'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        #13 + 3 = vgg 16
        'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        # 16 +3 =vgg 19
        'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
    }
    feature = Make_layers(cfg['D'], batch_norm=True)
    model = Vgg(feature, num_classes=7, init_weights=True)
    model.train()

#######train

    epoch = 10
    lr = 0.0004
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    start_time = time.time()
    for epoch in range(epoch):
        trainLoss = 0
        trainSize = 0
        trainCorrect = 0

        # train
        for batchIdx, data in enumerate(trainData):
            images, labels = data

            images, labels = images.to(device), labels.to(device)
            images = images.float()
            optimizer.zero_grad()
            outputs = model(images)


            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()
            trainLoss = loss.item()

            _, predicted = outputs.max(1)
            trainSize += labels.shape[0]
            trainCorrect += predicted.eq(labels.view_as(predicted)).sum().item()
            trainAccuracy = 100 * trainCorrect / trainSize

        torch.save(model, '../model/vgg16_epoch{}_batch{}_accuracy{:.3f}.pt'.format(epoch+1, batchIdx+1, trainAccuracy))
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': trainLoss,
        }, '../model/vgg16_epoch{}_batch{}_accuracy{:.3f}.pth'.format(epoch+1, batchIdx+1, trainAccuracy))

        #print('epoch {} batch {} train_loss {}  accuracy {}'.format(epoch+1, i+1, trainLoss / trainSize, trainAccuracy))
        print('Epoch {}/{}'.format(epoch, 10))
        print("---------")
        print('train Loss: {}  Acc: {}'.format(trainLoss / trainSize, trainAccuracy))
        trainLoss = 0.0
        # validation
        with torch.no_grad():
            valLoss = 0
            valSize = 0
            valCorrect = 0

            for batchIdx, data in enumerate(valData):
                images, labels = data
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                valLoss = criterion(outputs, labels).item()

                _, predicted = outputs.max(1)
                valSize += labels.shape[0]

                valCorrect += predicted.eq(labels.view_as(predicted)).sum().item()
                valAccuracy = 100 * valCorrect / valSize
            print('val Loss: {}  Acc: {}'.format(valLoss / valSize, valAccuracy))
            valLoss = 0.0

    # test
    model.eval()

    testLoss = 0
    testSize = 0
    testCorrect = 0

    with torch.no_grad():
        for batchIdx, (images, labels) in enumerate(testData):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            testLoss = criterion(outputs, labels).item()
            _, predicted = outputs.max(1)

            testSize += labels.shape[0]

            testCorrect += predicted.eq(labels.view_as(predicted)).sum().item()
            testLoss /= len(testData.dataset)

        accuracy = 100 * testCorrect / testSize
        print(accuracy)



if __name__ == "__main__":
    main()





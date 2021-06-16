import torch.optim as optim
import time
import logging   # 20210610
from tqdm import tqdm
import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
from configs import Configs
from logs import log
from vggModule import *
from MyDataLoader import *
from utils.wandb_logging.wandb_utils import WandbLogger, check_wandb_resume   # 20210610


#os.environ["CUDA_VISIBLE_DEVICES"]="1"


###### 20210610  #####################################
#configure the logging level to INFO   
#logging.basicConfig(level=logging.INFO)  # INFO 이상의 이벤트만 추적
#logger = logging.getLogger(__name__)   
######################################################

def train(model, trainloader, testloader):

    ############## 20210610 #########################
    # set a logger file
    logger = log(path="logs/", file="cross_val.logs")
    #################################################

    model.train()

    classes = ('fear', 'happy', 'sad', 'neutral', 'disgust', 'surprise', 'angry')
    import torch.optim as optim
    import time

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=Configs.lr)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logger.info("Epochs {}".format(Configs.epochs_nb))
    logger.info("Class Number {}".format(Configs.class_num))
    logger.info("LearningRate {}".format(Configs.lr))
    logger.info("ModelSize {}{}".format(Configs.model_name, Configs.model_config))


    start_time = time.time()
    for epoch in range(Configs.epochs_nb):  # loop over the dataset multiple times
        trainLoss = 0.0
        trainSize = 0
        trainCorrect = 0
        
        #with tqdm(total=100) as pbar:
        for i, data in enumerate(trainloader, 0):
            #time.sleep(0.1)
            #pbar.update(10)
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

        torch.save(model, '/data/FoodDetection/Object_Detection/yolov5-test/ssd/models/ep200_vgg11_dropout/vgg11_epoch{}_accuracy{:.3f}.pt'.format(epoch+1, trainAccuracy))
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': trainLoss,
            }, '/data/FoodDetection/Object_Detection/yolov5-test/ssd/models/ep200_vgg11_dropout/vgg11_epoch{}_accuracy{:.3f}.pth'.format(epoch+1, trainAccuracy))


        ############### 20210610 #################################################
        #print('epoch {} batch {} train_loss {}  accuracy {}'.format(epoch+1, i+1, trainLoss / trainSize, trainAccuracy))
        #print('\nEpoch {}/{}'.format(epoch+1, Configs.epochs_nb))
        #print("---------")
        #print('train Loss: {}  Acc: {:.3f}'.format(trainLoss / trainSize, trainAccuracy))
        logger.info("Train {}".format(model_name))
        logger.info('\nEpoch {}/{}'.format(epoch+1, Configs.epochs_nb))
        logger.info("---------------------")
        logger.info('train Loss: {}  Acc: {:.3f}'.format(trainLoss / trainSize, trainAccuracy))
        ###########################################################################
        trainLoss = 0.0 


            
            #if(loss.item() > 1000):
            #    print(loss.item())
            #    for param in model.parameters():
            #        print(param.data)
            
            ## print statistics
            ##trainLoss += loss.item()
            #if i % 50 == 49:    # print every 2000 mini-batches
            #    torch.save(model, '/content/drive/MyDrive/vgg16_epoch{}_batch{}_accuracy{:.3f}.pt'.format(epoch+1, i+1, trainAccuracy))
            #    print('epoch {} batch {} train_loss {}  accuracy {}'.format(epoch+1, i+1, trainLoss / 50, trainAccuracy))
            #    trainLoss = 0.0   # 이걸 해야할거같은..느낌...?
            

            
        model.eval()
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
            #print('val Loss: {}  Acc: {:.3f}'.format(valLoss / valSize, valAccuracy))    # 20210610
            logger.info('val Loss: {}  Acc: {:.3f}'.format(valLoss / valSize, valAccuracy))   # 20210610
            valLoss = 0.0



    #torch.save(model, '/content/drive/MyDrive/vgg16.pt')
    #torch.save(model.state_dict(), '/content/drive/MyDrive/vgg16.pth')
    print(int((time.time()-start_time)/60))
    print('Finished Training')


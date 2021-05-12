from glob import glob
import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from skimage import io
from torchvision.models.vgg import make_layers
import torchvision.transforms as transforms
import torch.nn as nn
import shutil
import os
import sys
from PIL import Image


# custum dataset
class MyFaceSet(Dataset):
    def __init__(self, data_path_list, classes, transform=None):
        self.path_list = data_path_list
        self.label = self.get_label(data_path_list)
        self.transform = transform
        self.classes = classes

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = io.imread(self.path_list[idx], as_gray=True)
        if self.transform is not None:
            image = self.transform(image)

        return image, self.classes.index(self.label[idx])

    def get_label(self, data_path_list):
        label_list = []
        for path in data_path_list:
            label_list.append(path.split('/')[-2])
 
        return label_list


# data loader
def dataloader(DATA_PATH_TRAINING_LIST, DATA_PATH_TESTING_LIST) :
    transform = transforms.Compose([transforms.ToTensor(),
                                   #transforms.Grayscale(),
                                   transforms.Resize((224,224)),
                                   transforms.Normalize((0.5,), (0.5,))])

    classes = ('angry','disgust','fear','happy','neutral','sad','surprise')
    trainloader = torch.utils.data.DataLoader(MyFaceSet(DATA_PATH_TRAINING_LIST,
                                                        classes,
                                                        transform=transform),
                                              batch_size=4,
                                              shuffle=True)

    testloader = torch.utils.data.DataLoader(MyFaceSet(DATA_PATH_TESTING_LIST,
                                                       classes,
                                                       transform=transform),
                                             batch_size=4,
                                             shuffle=False )

    return trainloader, testloader


# data split
def split_dataset_into_3(path_to_dataset, train_ratio, valid_ratio):
    _, sub_dirs, _ = next(iter(os.walk(path_to_dataset)))
    sub_dir_item_cnt = [0 for i in range(len(sub_dirs))]

    # directories where the splitted dataset will lie
    dir_train = os.path.join(os.path.dirname(path_to_dataset), 'train')
    dir_valid = os.path.join(os.path.dirname(path_to_dataset), 'validation')
    dir_test = os.path.join(os.path.dirname(path_to_dataset), 'test')

    for i, sub_dir in enumerate(sub_dirs):
        print(i,sub_dir)
        dir_train_dst = os.path.join(dir_train, sub_dir)
        dir_valid_dst = os.path.join(dir_valid, sub_dir)
        dir_test_dst = os.path.join(dir_test, sub_dir)

        # variables to save the sub directory name(class name) and to count the images of each sub directory(class)
        class_name = sub_dir
        sub_dir = os.path.join(path_to_dataset, sub_dir)
        sub_dir_item_cnt[i] = len(os.listdir(sub_dir))

        items = os.listdir(sub_dir)

        # transfer data to trainset
        for item_idx in range(round(sub_dir_item_cnt[i] * train_ratio)):
            if not os.path.exists(dir_train_dst):
                os.makedirs(dir_train_dst)

            source_file = os.path.join(sub_dir, items[item_idx])
            dst_file = os.path.join(dir_train_dst, items[item_idx])
            shutil.copyfile(source_file, dst_file)

        # transfer data to validation
        for item_idx in range(round(sub_dir_item_cnt[i] * train_ratio) + 1,
                              round(sub_dir_item_cnt[i] * (train_ratio + valid_ratio))):
            if not os.path.exists(dir_valid_dst):
                os.makedirs(dir_valid_dst)

            source_file = os.path.join(sub_dir, items[item_idx])
            dst_file = os.path.join(dir_valid_dst, items[item_idx])
            shutil.copyfile(source_file, dst_file)

        # transfer data to testset
        for item_idx in range(round(sub_dir_item_cnt[i] * (train_ratio + valid_ratio)) + 1, sub_dir_item_cnt[i]):
            if not os.path.exists(dir_test_dst):
                os.makedirs(dir_test_dst)

            source_file = os.path.join(sub_dir, items[item_idx])
            dst_file = os.path.join(dir_test_dst, items[item_idx])
            shutil.copyfile(source_file, dst_file)


# vgg net
class Vgg(nn.Module):
    def __init__(self, features, num_classes=10, init_weights=True):
        super(Vgg, self).__init__()
        self.features = features  
        self.avgpool = nn.AvgPool2d(7)
        self.classifier = nn.Linear(512, 7)

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


# make layer
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


# model setting
path = './drive/Shareddrives/zeogi_gogi/dataset/'
DATA_PATH_TRAINING_LIST = glob(path+'train/*/*.jpg')
DATA_PATH_TESTING_LIST = glob(path+'test/*/*.jpg')

trainloader, testloader = dataloader(DATA_PATH_TRAINING_LIST, DATA_PATH_TESTING_LIST)
dataiter = iter(trainloader)
images, labels = dataiter.next()
cfg = { #8 + 3 =11 == vgg11
        'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        # 10 + 3 = vgg 13
        'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        #13 + 3 = vgg 16
        'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        # 16 +3 =vgg 19
        'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'] }

feature = make_layers(cfg['A'], batch_norm=True)
model = Vgg(feature, num_classes=7, init_weights=True)
model


# train
model.train()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
epoch = 10
lr = 0.0004
optimizer = torch.optim.Adam(model.parameters(), lr = lr)
criterion = nn.CrossEntropyLoss()

for epoch in range(2):
    trainLoss = 0.0
    trainSize = 0.0
    trainCorrect = 0.0
    trainAccuracy = 0.0

    # train 
    for batchIdx, data in enumerate(trainloader):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        images = images.float()
    
        optimizer.zero_grad()
        model.cuda()
        outputs = model(images)
    
        loss = criterion(outputs, labels)
        loss.backward()
    
        optimizer.step()
        trainLoss = loss.item()
    
        _, predicted = outputs.max(1)
        trainSize += labels.shape[0]
        trainCorrect += predicted.eq(labels.view_as(predicted)).sum().item()
        trainAccuracy = 100 * trainCorrect / trainSize

    print(epoch, 'epoch, training acc: ', trainAccuracy, ',training loss: ', trainLoss)

    """
    # validation
    with torch.no_grad():
        valLoss = 0.0
        valSize = 0.0
        valCorrect = 0.0

        for batchIdx, data in enumerate(valData):
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            valLoss = criterion(outputs, labels).item()

            _, predicted = outputs.max(1)
            valSize += labels.shape[0]

            valCorrect += predicted.eq(labels.view_as(predicted)).sum().item()
            valAccuracy = 100 * valCorrect / valSize
    """


# test
model.eval()

testLoss = 0.0
testSize = 0.0
testCorrect = 0.0

with torch.no_grad():
    for batchIdx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)
        images = images.float()
        
        model.cuda()
        outputs = model(images)

        testLoss = criterion(outputs, labels).item()
        _, predicted = outputs.max(1)

        testSize += labels.shape[0]

        testCorrect += predicted.eq(labels.view_as(predicted)).sum().item()
        testLoss /= len(testloader.dataset)

    accuracy = 100 * testCorrect / testSize

    print('testing acc: ', accuracy, ',testing loss: ', testLoss)



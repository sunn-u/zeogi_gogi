from torch.utils.data import *
from torchvision import transforms
import torch
#from torch.utils.data import Dataset, DataLoader, ConcatDataset, Sampler, RandomSampler, BatchSampler
from torch.utils.data import *
from skimage import io, transform
from torchvision import transforms
import cv2
from PIL import Image

from sklearn.model_selection import train_test_split
from glob import glob
import os
from configs import Configs


classes = ('fear', 'happy', 'sad', 'neutral', 'disgust', 'surprise', 'angry')


class MyDataSet(Dataset):
    #data_path_list - 이미지 path 전체 리스트
    #label - 이미지 ground truth
    def __init__(self, data_path_list, classes, transform=None):
        self.path_list = data_path_list
        self.label = MyDataSet.get_label(data_path_list)
        self.transform = transform
        self.classes = classes

    def get_label(data_path_list):
        label_list = []
        for path in data_path_list:
            # 뒤에서 두번째가 class다.
            label_list.append(path.split('/')[-2])   
        return label_list

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = io.imread(self.path_list[idx], as_gray=True)     # 이미지가 grayscale과 컬러가 섞여있어 gray로 통일
        if self.transform is not None:
            image = Image.fromarray(image)
            image = self.transform(image)
            return image, self.classes.index(self.label[idx])



def make_train_val_set(data_path):
    d_path = data_path
    #f_dir = os.listdir(d_path)
    f_dir = ['fear', 'happy', 'sad', 'neutral', 'disgust', 'surprise', 'angry']

    TRAIN_LIST = []
    VAL_LIST = []

    for i in f_dir:
        DATA_LIST = glob(d_path + str(i) + '/*.jpg')
        train_img_list, val_img_list = train_test_split(DATA_LIST[:4000], test_size = 0.1, random_state=2002)
        print(len(train_img_list), len(val_img_list))
        for j in train_img_list:
            TRAIN_LIST.append(j)
        for k in val_img_list:
            VAL_LIST.append(k)

    transform = transforms.Compose(
        [transforms.Resize((224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])


    #own dataset
    trainloader = torch.utils.data.DataLoader(MyDataSet(TRAIN_LIST, classes,transform=transform),batch_size=64,shuffle = True,drop_last=True)
    testloader = torch.utils.data.DataLoader(MyDataSet(VAL_LIST, classes,transform=transform),batch_size=64,shuffle = False,drop_last=True)

    return trainloader, testloader

      

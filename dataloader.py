from glob import glob

import torch
from torchvision import transforms, datasets
import torchvision
import sys

DATA_PATH_TRAINING_LIST = glob('/Users/ahrim/Desktop/dataset/train/*/*.jpg')
DATA_PATH_TESTING_LIST = glob('/Users/ahrim/Desktop/dataset/test/*/*.jpg')

from torch.utils.data import Dataset, DataLoader
from skimage import io

transform = transforms.Compose(
    [transforms.ToTensor(),
     #transforms.Grayscale(),
     transforms.Resize((224,224)),
     transforms.Normalize((0.5,), (0.5,))]
)


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
        image = io.imread(self.path_list[idx], as_gray=True) ## 차원 맞추
        if self.transform is not None:
            image = self.transform(image)
        print(self.classes)
        # print(self.label)
        return image, self.classes.index(self.label[idx])

    def get_label(self, data_path_list):
        label_list = []
        for path in data_path_list:
            label_list.append(path.split('/')[-2])
            # print(path)
            # print(path.split('\\')[-2])
        return label_list




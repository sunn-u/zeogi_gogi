import torch
#from torch.utils.data import Dataset, DataLoader, ConcatDataset, Sampler, RandomSampler, BatchSampler
from torch.utils.data import *
from skimage import io, transform
from torchvision import transforms
import cv2

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
      image = self.transform(image)
      return image, self.classes.index(self.label[idx])



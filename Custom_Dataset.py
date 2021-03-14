
import os
from PIL import Image
from glob import glob
import random
from collections import defaultdict
from torch.utils.data import Dataset

from configs import Configs


class CustomDataset(Dataset):
    def __init__(self, data_root, transform=None):
        self.data_list = self.get_images(data_root)
        self.label_list, self.label_dict = self.get_labels()
        self.classes = list(self.label_dict.keys())
        self.transform = transform

        self.color = 'L' if Configs.image_gray else 'RGB'
        self.split = Configs.split

    def get_images(self, data_root):
        img_list = []
        extension = ['*/*.jpg', '*/*.png', '*/*.gif', '*/*.jpeg', '*/*.bmp']
        for ext in extension:
            img_list += glob(os.path.join(data_root, ext))

        return img_list

    def get_labels(self):
        label_list = []
        label_dict = defaultdict(list)
        for img_path in self.data_list:
            label = img_path.split('\\')[-2]
            label_list.append(label)
            label_dict[label].append(img_path)

        return label_list, label_dict

    # not for use
    def split_sets(self):
        val_percent = 1 - Configs.train_percent

        train_data, train_label = [], []
        val_data, val_label = [], []
        for key, values in self.label_dict.items():
            train_cnt = int(len(values)*Configs.train_percent)
            val_cnt = int(len(values)*val_percent)

            train_data += random.sample(values, train_cnt)
            train_label += [key]*train_cnt
            rest = list(set(values) - set(train_data))
            val_data += random.choice(rest, val_cnt)
            val_label += [key]*val_cnt

        return train_data, train_label, val_data, val_label

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        image = Image.open((self.data_list[idx]))
        image = image.convert(self.color)

        if self.transform is not None:
            image = self.transform(image)

        return image, self.label_list[idx]

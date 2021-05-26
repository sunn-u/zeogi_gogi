
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

from VggModule import VGG
from MyFaceLoader import MyFaceSet
from utils import get_images


class Build:
    def __init__(self, args):
        self.args = args

    def build_model(self):
        model = VGG(model_name=self.args.model_name, num_classes=len(self.args.classes), init_weights=True)
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        criterion = torch.nn.CrossEntropyLoss()

        return model, optimizer, criterion

    def build_loader(self):
        data_trans = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, ))
        ])

        train_data_list, val_data_list = get_images(self.args.data_dir, self.args.classes)

        train_set = MyFaceSet(train_data_list, self.args.classes, transform=data_trans)
        val_set = MyFaceSet(val_data_list, self.args.classes, transform=data_trans)

        train_loader = DataLoader(train_set, batch_size=self.args.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=self.args.batch_size, shuffle=True, drop_last=True)

        return train_loader, val_loader

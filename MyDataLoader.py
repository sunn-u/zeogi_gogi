from torch.utils.data import *
from torchvision import transforms

from MyDataSet import MyDataset


def make_train_val_set(data_path):
      d_path = data_path
      f_dir = os.listdir(d_path)

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

      return TRAIN_LIST, VAL_LIST

      

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((200,200))])


#own dataset
trainloader = torch.utils.data.DataLoader(MyDataSet(TRAIN_LIST, classes,transform=transform),batch_size=4,shuffle = True,drop_last=True)
testloader = torch.utils.data.DataLoader(MyDataSet(VAL_LIST, classes,transform=transform),batch_size=4,shuffle = False,drop_last=True)
from torchvision.models.vgg import make_layers

import splitdata
from dataloader import *
import matplotlib.pyplot as plt
import numpy as np

from vggModule import VGG


def dataloader() :
    # splitdata.split_dataset_into_3('/Users/ahrim/Desktop/dataset', 0.7, 0)

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
    return trainloader, testloader

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

def main():
    trainloader, testloader = dataloader()
    dataiter = iter(trainloader)


    cfg = {
        #8 + 3 =11 == vgg11
        'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        # 10 + 3 = vgg 13
        'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        #13 + 3 = vgg 16
        'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        # 16 +3 =vgg 19
        'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'] }


    feature = make_layers(cfg['A'], batch_norm=True)

    CNN = VGG(feature, num_classes=10, init_weights=True)

    CNN
    # images, labels = dataiter.next()
    # print(images, labels)
    # imshow(torchvision.utils.make_grid(images))

    # print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

if __name__ == "__main__":
    main()





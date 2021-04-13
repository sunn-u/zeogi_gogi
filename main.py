import splitdata
from dataloader import *
import matplotlib.pyplot as plt
import numpy as np


def dataloader() :
    splitdata.split_dataset_into_3('./dataset/', 0.7, 0)

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

    images, labels = dataiter.next()
    print(images, labels)
    imshow(torchvision.utils.make_grid(images))

    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

if __name__ == "__main__":
    main()





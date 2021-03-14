
from torchvision.utils import save_image
from Data_Loader import train_loader


def main():
    data_iter = iter(train_loader)
    for idx, (images, labels) in enumerate(data_iter):
        save_image(images, f'{idx}_save_test.jpg')


if __name__ == '__main__':
    main()
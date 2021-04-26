from torchvision.utils import save_image
from Data_Loader import train_loader
from MyDataLoader import make_train_val_set


def main():
	make_train_val_set(opt.data)
	dataiter = iter(trainloader)
	images, labels = dataiter.next()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data', type=str, default='/content/gdrive/Shareddrives/zeogi_gogi/dataset/', help='dataset path')
	opt = parser.parse_args()
	
	main()
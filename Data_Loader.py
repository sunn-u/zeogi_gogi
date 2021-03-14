
from torchvision import transforms
from torch.utils.data import DataLoader

from Custom_Dataset import CustomDataset
from configs import Configs


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((200, 200)),
     transforms.Normalize(Configs.normalize, Configs.normalize)])

train_loader = DataLoader(
    CustomDataset(
        Configs.data_root,
        transform=transform),
    batch_size=5,
    shuffle=True,
    drop_last=True)

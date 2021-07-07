import os
import torch
import torch.optim as optim
import argparse
import torchvision.models as models
from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader

from utils import get_images, set_log
from MyFaceLoader import MyFaceSet


def set_args():
    parser = argparse.ArgumentParser(description='setting for emotion recognition')

    # fixing arguments
    parser.add_argument('--classes', default=['disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'])
    parser.add_argument('--output_dir', default='')
    parser.add_argument('--epochs', default=300)
    parser.add_argument('--batch_size', default=16)

    # flexible arguments
    parser.add_argument('--input_dir', type=str)
    parser.add_argument('--expr_name', type=str)
    parser.add_argument('--model_name', type=str)

    return parser.parse_args()


def prepare_dataset(opt):
    data_trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_data_list, val_data_list = get_images(opt.input_dir, opt.classes)

    train_set = MyFaceSet(train_data_list, opt.classes, transform=data_trans)
    val_set = MyFaceSet(val_data_list, opt.classes, transform=data_trans)

    train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=opt.batch_size, shuffle=True, drop_last=True)

    return train_loader, val_loader


def call_models(model_name, pre_trained=True):
    model_dict = {
        'resnet18': models.resnet18(pretrained=pre_trained),
        'alexnet': models.alexnet(pretrained=pre_trained),
        'squeezenet': models.squeezenet1_0(pretrained=pre_trained),
        'vgg16': models.vgg16(pretrained=pre_trained),
        'densenet': models.densenet161(pretrained=pre_trained),
        'inception':  models.inception_v3(pretrained=pre_trained),
        'googlenet': models.googlenet(pretrained=pre_trained),
        'shufflenet':  models.shufflenet_v2_x1_0(pretrained=pre_trained),
        'mobilenet_v2':  models.mobilenet_v2(pretrained=pre_trained),
        'resnext50_32x4d': models.resnext50_32x4d(pretrained=pre_trained),
        'wide_resnet50_2': models.wide_resnet50_2(pretrained=pre_trained),
        'mnasnet': models.mnasnet1_0(pretrained=pre_trained)
    }

    return model_dict[model_name]


def run_main():
    opt = set_args()

    opt.input_dir = '/workspace/codes/sunn/tmp/dataset'
    opt.output_dir = '/workspace/codes/sunn/tmp/results'

    opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir = os.path.join(opt.output_dir, opt.expr_name)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger = set_log(logger_name=f'{opt.expr_name}_log', file_dir=f'{output_dir}/log.txt')

    train_loader, val_loader = prepare_dataset(opt)
    model = call_models(model_name=opt.model_name, pre_trained=True)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.CrossEntropyLoss()
    model = model.to(opt.device)

    best_score = 0
    # training time
    for epoch in range(opt.epochs):
        model.train()
        trainLoss, trainCorrect = 0, 0

        for idx, data in enumerate(train_loader):
            images, targets = data[0].to(opt.device), data[1].to(opt.device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            _, predicted = outputs.max(1)
            trainLoss += loss.item()
            trainCorrect += predicted.eq(targets.view_as(predicted)).sum().item()

        TrainAccuracy = 100 * trainCorrect / len(train_loader.dataset)
        meanLosss = 100 * trainLoss / len(train_loader.dataset)
        logger.debug(f'[{epoch}/{opt.epochs}] Train Accuracy : {TrainAccuracy}, Train Loss : {meanLosss}')

        if epoch % 2 == 0:
            model.eval()
            valLoss, valCorrect = 0, 0

            for idx, data in enumerate(val_loader):
                images, targets = data[0].to(opt.device), data[1].to(opt.device)

                with torch.no_grad():
                    outputs = model(images)
                    loss = criterion(outputs, targets)

                    _, predicted = outputs.max(1)
                    valLoss += loss.item()
                    valCorrect += predicted.eq(targets.view_as(predicted)).sum().item()

            ValAccuracy = 100 * valCorrect / len(val_loader.dataset)
            meanLosss = 100 * valLoss / len(val_loader.dataset)
            logger.debug(f'[{epoch}/{opt.epochs}] Val Accuracy : {ValAccuracy}, Val Loss : {meanLosss}')

            if ValAccuracy > best_score:
                best_score = ValAccuracy
                torch.save(model.state_dict(), os.path.join(output_dir, f'{epoch}_bestmodel.pth'))


if __name__ == '__main__':
    run_main()

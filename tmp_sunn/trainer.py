
import os
import torch
from pathlib import Path

from build import Build
from utils import set_logger


class Trainer:
    def __init__(self, args, device):
        self.args = args
        self.device = device

        Path(self.args.output_dir).mkdir(parents=True, exist_ok=True)

    def _settings(self):
        building = Build(self.args)
        model, optimizer, criterion = building.build_model()
        train_loader, val_loader = building.build_loader()

        return model, optimizer, criterion, train_loader, val_loader

    # todo : item 뭐지?
    def run(self):
        # setting data & model
        model, optimizer, criterion, train_loader, val_loader = self._settings()
        model = model.to(self.device)
        best_score = 0

        # training time
        for epoch in range(self.args.epochs):
            model.train()
            trainLoss, trainCorrect = 0, 0

            for idx, data in enumerate(train_loader):
                images, targets = data[0].to(self.device), data[1].to(self.device)

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
            print(f'[{epoch}/{self.args.epochs}] Train Accuracy : {TrainAccuracy}, Train Loss : {meanLosss}')

            if epoch % 10 == 0:
                model.eval()
                valLoss, valCorrect = 0, 0

                for idx, data in enumerate(val_loader):
                    images, targets = data[0].to(self.device), data[1].to(self.device)

                    with torch.no_grad():
                        outputs = model(images)
                        loss = criterion(outputs, targets)

                        _, predicted = outputs.max(1)
                        valLoss += loss.item()
                        valCorrect += predicted.eq(targets.view_as(predicted)).sum().item()

                ValAccuracy = 100 * valCorrect / len(val_loader.dataset)
                meanLosss = 100 * valLoss / len(val_loader.dataset)
                print(f'[{epoch}/{self.args.epochs}] Val Accuracy : {ValAccuracy}, Train Loss : {meanLosss}')

                if ValAccuracy > best_score:
                    best_score = ValAccuracy
                    torch.save(model.state_dict(), os.path.join(self.args.output_dir, f'{epoch}_bestmodel.pth'))

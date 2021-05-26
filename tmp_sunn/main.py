
import torch
import argparse

from trainer import Trainer


def set_args():
    parser = argparse.ArgumentParser(description='setting for emotion recognition')

    parser.add_argument('--data_dir', default='/home/infiniq/sunn/tmptmp/dataset')
    parser.add_argument('--output_dir', default='/home/infiniq/sunn/tmptmp/results/0526')
    parser.add_argument('--classes', default=['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'])
    parser.add_argument('--epochs', default=10)
    parser.add_argument('--batch_size', default=64)
    parser.add_argument('--model_name', default='VGG16', help='choose model with number - 11, 13, 16, 19')
    parser.add_argument('--model_config', default='D', help='')

    return parser.parse_args()


def run_main():
    global args
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = Trainer(args, device)
    trainer.run()


if __name__ == '__main__':
    args = set_args()
    run_main()

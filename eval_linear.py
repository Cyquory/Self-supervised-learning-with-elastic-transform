import argparse
import os
import logging
import time
from pathlib import Path

import numpy as np
from PIL import Image
from collections import OrderedDict

def get_args_parser():
    parser = argparse.ArgumentParser(description='workflow for self-supervised learning')

    # Model parameters
    parser.add_argument('--image_size', default=224, type=int, help='Size in pixels of input image')

    # Training/Optimization parameters
    parser.add_argument('--batch_size_per_gpu', default=64, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=10, type=int, help='Number of epochs of training.')
    parser.add_argument('--num_labels', default=1000, type=int, help='Number of labels for linear classifier')

    # Misc
    parser.add_argument('--cuda', default='0', type=str, help='GPU that can be seen by the models')
    parser.add_argument('--data_path', default='/path/to/imagenet/', type=str,
        help='Please specify path to the ImageNet root folder.')
        # /tmp2/dataset/imagenet/ILSVRC/Data/CLS-LOC/
    parser.add_argument('--ckpt_path', default='/path/to/checkpoint/', type=str,
        help='Please specify path to the pretext model checkpoints.')
        # /tmp2/aislab/ckpt/SSL/Epoch_0005.pth
    parser.add_argument('--seed', default=3084, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=12, type=int, help='Number of data loading workers per GPU.')
    
    return parser

args = get_args_parser().parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
# pytorch related package 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.utils import save_image

import lib.utils

print('pytorch version: ' + torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def eval_linear():
    # ============ setup environment ============
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # ============ building network ... ============
    arch = 'resnet50'
    model = models.__dict__[arch]()
    in_feature = model.fc.in_features
    model.fc = nn.Identity()
    ckpt = torch.load(args.ckpt_path)
    model_state_dict = ckpt['model_state_dict']
    model_state_dict = {k.replace("fc.", ""): v for k, v in model_state_dict.items()}
    model.load_state_dict(model_state_dict, strict=False)
    model.to(device)
    model.eval()

    linear_classifier = LinearClassifier(in_feature, num_labels=args.num_labels)
    linear_classifier.to(device)
    linear_classifier.train()

    # ============ preparing data ... ============
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std  = (0.229, 0.224, 0.225)

    val_transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode('bicubic')),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])
    dataset_val = datasets.ImageFolder(root=Path(args.data_path)/'val', transform=val_transform)
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size_per_gpu, \
        shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True,)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])
    dataset_train = datasets.ImageFolder(Path(args.data_path)/'train', transform=train_transform)
    train_loader = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size_per_gpu, \
        shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True,)
    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

    # ============ preparing loss and optimizer ============
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(linear_classifier.parameters())

    # ============ training ... ============
    def train():
        train_loss, correct, total = 0, 0, 0

        for batch_idx, (data, label) in enumerate(train_loader):
            data, label = data.to(device), label.to(device)

            # train the model
            optimizer.zero_grad()
            output = model(data)
            logit = linear_classifier(output)
            loss = criterion(logit, label)
            loss.backward()
            optimizer.step()
            preds = F.softmax(logit, dim=1)
            preds_top_p, preds_top_class = preds.topk(1, dim=1)
            train_loss += loss.item() * label.size(0)
            total += label.size(0)
            correct += (preds_top_class.view(label.shape) == label).sum().item()

            if batch_idx > 1000:
                print('==> early break in training')
                break
            
        return (train_loss / batch_idx, 100. * correct / total)

    start_time = time.time()
    print("Starting supervised linear classifier training !")
    start_epoch = 0
    for epoch in range(start_epoch, args.epochs):
        train_loss, train_acc = train()
        print('train loss:{:.4f}, train acc:{:.4f}'.format(train_loss, train_acc))

    total_time = time.time() - start_time
    print(f'Training time {total_time/60:.2f} minutes')


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)

if __name__ == '__main__':
    eval_linear()
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from lib import torch_util, storage
from .base import RecognizerBase


class OneShotNetwork(nn.Module):
    def __init__(self):
        super(OneShotNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),

            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

        )

        self.fc1 = nn.Sequential(
            nn.Linear(8 * 100 * 100, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 5)
        )

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) + label * torch.pow(
            torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


class NetworkDataSet(Dataset):
    def __init__(self, data_folder, transform=None, should_invert=True):
        self.data_folder = data_folder
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self, index):
        img0_tuple = random.choice(self.data_folder.imgs)
        should_get_same_class = random.randint(0, 1)
        if should_get_same_class:
            while True:
                img1_tuple = random.choice(self.data_folder.imgs)
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:
            while True:
                img1_tuple = random.choice(self.data_folder.imgs)
                if img0_tuple[1] != img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        img0 = img0.convert("L")
        img1 = img1.convert("L")

        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))

    def __len__(self):
        return len(self.data_folder.imgs)


class Recognizer(RecognizerBase):
    def __init__(self, rec_type, config):
        super(Recognizer, self).__init__(rec_type, config)

        default_conf = {'lr': 0.0002, 'num_epochs': 150, 'batch_size': 20}
        self.config = {**default_conf, **config}

        self.sample_path = storage.get_local_path(storage.TYPE_SAMPLE, self.rec_type)
        self.train_sample_path = '{}/train'.format(self.sample_path)
        self.val_sample_path = '{}/val'.format(self.sample_path)

        self.model_path = storage.download(storage.TYPE_MODEL, self.config.get('pretrained_model_name'))

    def recognize(self, image_str):
        return {
            'labels': torch_util.predict_str(self.model, self.labels, self.transform, image_str)
        }

    def train(self):
        net = OneShotNetwork().cuda()
        criterion = ContrastiveLoss()
        optimizer = optim.Adam(net.parameters(), lr=self.config.get('lr'))

        folder_train_set = dset.ImageFolder(root=self.train_sample_path)
        siamese_data_set = NetworkDataSet(
            data_folder=folder_train_set,
            transform=transforms.Compose([transforms.Resize((100, 100)), transforms.ToTensor()]),
            should_invert=False
        )
        train_data_loader = DataLoader(
            siamese_data_set,
            shuffle=True,
            num_workers=8,
            batch_size=self.config.get('batch_size')
        )

        iteration_number = 0
        for epoch in range(0, self.config.get('num_epochs')):
            for i, data in enumerate(train_data_loader, 0):
                img0, img1, label = data
                img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()
                optimizer.zero_grad()
                output1, output2 = net(img0, img1)
                loss_contrastive = criterion(output1, output2, label)
                loss_contrastive.backward()
                optimizer.step()
                if i % 10 == 0:
                    print("Epoch number {}\n Current loss {}\n".format(epoch, loss_contrastive.item()))
                    iteration_number += 10

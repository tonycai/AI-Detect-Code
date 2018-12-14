# -*- coding:utf-8 -*-

import time
import os
import copy
import io
import sys

from PIL import Image

import torch

from torchvision import datasets

import torchnet.meter as meter

device_id = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_id)


def get_dataloader(folder, transform, batch_size=100, num_workers=4):
    dataset = datasets.ImageFolder(folder, transform)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


def train_model(model, criterion, optimizer, scheduler, train_dataloader, val_dataloader, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        train_model_epoch(model, criterion, optimizer, scheduler, train_dataloader)
        val_loss, val_acc = validate_model(model, criterion, val_dataloader, train_dataloader.dataset.classes)
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)


def train_model_epoch(model, criterion, optimizer, scheduler, dataloader):
    model.train()  # Set model to training mode
    scheduler.step()

    total_loss = 0.0
    total_corrects = 0
    total_size = 0

    # Iterate over data.
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # statistics
        total_loss += loss.item() * inputs.size(0)
        total_corrects += torch.sum(preds == labels.data)
        total_size += inputs.size(0)

    loss = total_loss / total_size
    acc = total_corrects.double() / total_size

    print('Train Loss: {:.4f} Acc: {:.4f}'.format(loss, acc))

    return loss, acc


def validate_model(model, criterion, dataloader, all_labels, confusion_matrix_enable=True):
    model.eval()  # Set model to evaluate mode

    total_loss = 0.0
    total_corrects = 0
    total_size = 0
    confusion_matrix = meter.ConfusionMeter(len(all_labels))

    # Iterate over data.
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            if confusion_matrix_enable:
                confusion_matrix.add(outputs, labels)

        # statistics
        total_loss += loss.item() * inputs.size(0)
        total_corrects += torch.sum(preds == labels.data)
        total_size += inputs.size(0)

    loss = total_loss / total_size
    acc = total_corrects.double() / total_size

    print('Val Loss: {:.4f} Acc: {:.4f}'.format(loss, acc))
    if confusion_matrix_enable:
        print(all_labels)
        print(confusion_matrix.conf)

    return loss, acc


def dump_model(model):
    return {
        'state_dict': model.state_dict()
    }


def recover_model(model, dump):
    model.load_state_dict(dump['state_dict'])
    return model


def store_data(data, file_path):
    torch.save(data, file_path)


def load_data(file_path):
    return torch.load(file_path, map_location=device_id)


def predict(model, labels, img_tensor):
    model.eval()

    img_tensor.unsqueeze_(0)

    img_tensor = img_tensor.to(device)

    outputs = model(img_tensor)

    return {label: float(torch.exp(outputs[0][i])) for i, label in enumerate(labels)}


def predict_str(model, labels, transform, img_str):
    img_pil = Image.open(io.BytesIO(img_str))
    img_pil = img_pil.convert('RGB')

    return predict(model, labels, transform(img_pil))


def predict_file(model, labels, transform, img_path):
    with open(img_path, 'rb') as f:
        img_pil = Image.open(f)
        img_pil = img_pil.convert('RGB')

    return predict(model, labels, transform(img_pil))


def predict_folder(model, labels, transform, img_folder, expected_label):
    i = 0
    for name in os.listdir(img_folder):
        i += 1
        sys.stdout.write('%d \r' % i)
        sys.stdout.flush()

        img_path = img_folder + '/' + name
        probs = predict_file(model, labels, transform, img_path)
        label = max(probs, key=probs.get)
        if label != expected_label:
            print(label, probs, img_path)

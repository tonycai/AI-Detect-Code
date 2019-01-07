# -*- coding:utf-8 -*-

from lib import torch_util, torch_model, storage
from .base import TorchBase
from torchvision import models
from model.error import ModelError


class Recognizer(TorchBase):

    def _get_model(self, labels_count):
        model_type = self.config.get('model_type')
        if model_type == 34:
            model = models.resnet34(pretrained=True)
        elif model_type == 18:
            model = models.resnet18(pretrained=True)
        elif model_type == 50:
            model = models.resnet50(pretrained=True)
        elif model_type == 152:
            model = models.resnet152(pretrained=True)
        elif model_type == 101:
            model = models.resnet101(pretrained=True)
        else:
            raise ModelError('invalid model_type')

        type_model_map = {
            18: 'resnet18-5c106cde.pth',
            34: 'resnet34-333f7ec4.pth',
            50: 'resnet50-19c8e357.pth',
            101: 'resnet101-5d3b4d8f.pth',
            152: 'resnet152-b121ed2d.pth'
        }
        pre_trained_model = storage.download(storage.TYPE_MODEL, type_model_map.get(model_type))
        model.load_state_dict(torch_util.load_data(pre_trained_model))

        if self.config.get('fixed_param'):
            for param in model.parameters():
                param.requires_grad = False

        model.fc = torch_model.ProbNet(model.fc.in_features, labels_count)

        model = model.to(torch_util.device)

        return model

# -*- coding:utf-8 -*-

from lib import torch_util, torch_model, storage
from .base import TorchBase
from torchvision import models
from model.error import ModelError


class Recognizer(TorchBase):

    def _get_model(self, labels_count):
        model_type = self.config.get('model_type')
        if model_type == 121:
            model = models.densenet121(pretrained=True)
        elif model_type == 169:
            model = models.densenet169(pretrained=True)
        elif model_type == 161:
            model = models.densenet161(pretrained=True)
        elif model_type == 201:
            model = models.densenet201(pretrained=True)
        else:
            raise ModelError('非法的模型类型')

        type_model_map = {
            121: 'densenet121-a639ec97.pth',
            169: 'densenet169-b2777c0a.pth',
            201: 'densenet201-c1103571.pth',
            161: 'densenet161-8d451a50.pth',
        }
        pre_trained_model = storage.download(storage.TYPE_MODEL, type_model_map.get(model_type))
        model.load_state_dict(torch_util.load_data(pre_trained_model))

        if self.config.get('fixed_param'):
            for param in model.parameters():
                param.requires_grad = False

        model.fc = torch_model.ProbNet(model.fc.in_features, labels_count)

        return model.to(torch_util.device)

# -*- coding:utf-8 -*-

import time
from torchvision import transforms
from lib import torch_util, torch_model, storage
from .base import TorchBase


class Recognizer(TorchBase):

    def _get_model(self, labels_count):
        model = models.inception_v3(pretrained=True)
        pre_trained_model = storage.download(storage.TYPE_MODEL, type_model_map.get(model_type))
        model.load_state_dict(torch_util.load_data(pre_trained_model))

        if self.config.get('fixed_param'):
            for param in model.parameters():
                param.requires_grad = False

        model.fc = torch_model.ProbNet(model.fc.in_features, labels_count)

        model = model.to(torch_util.device)

        return model

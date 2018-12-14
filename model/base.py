# -*- coding:utf-8 -*-

import time
from torchvision import transforms
from lib import torch_util, torch_model, storage


class RecognizerBase(object):

    def __init__(self, rec_type, config):
        self.rec_type = rec_type
        self.config = config

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def recognize(self, image_str):
        pass


class TorchBase(RecognizerBase):

    def __init__(self, rec_type, config):
        super(TorchBase, self).__init__(rec_type, config)
        self.sample_path = storage.get_local_path(storage.TYPE_SAMPLE, self.rec_type)
        self.train_sample_path = '{}/train'.format(self.sample_path)
        self.val_sample_path = '{}/val'.format(self.sample_path)

        if not config.get('train', False):
            model_path = storage.download(storage.TYPE_MODEL, self.rec_type, self.config.get('version_in_use'))
            model_data = torch_util.load_data(model_path)
            self.labels = model_data['labels']
            self.version = model_data['version']
            self.model = torch_util.recover_model(self._get_model(len(self.labels)), model_data['model'])
            self.transform = self._get_val_transform()

    def train(self):
        train_transform = self._get_train_transform()
        train_data_loader = torch_util.get_data_loader(self.train_sample_path, train_transform)

        labels = train_data_loader.dataset.classes

        val_transform = self._get_val_transform()
        val_data_loader = torch_util.get_data_loader(self.val_sample_path, val_transform)

        num_epochs = self.config.get('num_epochs', 10)

        criterion = nn.NLLLoss()

        # Observe that all parameters are being optimized
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, momentum=0.9)

        # Decay LR by a factor of 0.1 every 7 epochs
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        torch_util.train_model(
            self._get_model(len(labels)),
            criterion,
            optimizer,
            scheduler,
            train_data_loader,
            val_data_loader,
            num_epochs=num_epochs
        )

        model_data = {
            'labels': labels,
            'version': int(time.time()),
            'model': torch_util.dump_model(conf['model'])
        }
        model_path = storage.get_local_path(storage.TYPE_MODEL, self.rec_type, model_data['version'])
        torch_util.store_data(model_data, model_path)

    def recognize(self, image_str):
        return {
            'labels': torch_util.predict_str(self.model, self.labels, self.transform, image_str)
        }

    @abstractmethod
    def _get_model(self, labels_count):
        pass

    @classmethod
    def _get_train_transform(cls):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    @classmethod
    def _get_val_transform(cls):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

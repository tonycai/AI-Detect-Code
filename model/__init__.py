# -*- coding:utf-8 -*-

import sys

from . import resnet, inception_v3, dense_net
from .error import ModelError
from lib import storage

_recognizer_cache = {}

_rec_config = {
    'AJ#insole': {  # 鞋垫
        'model_type': 'resnet',
        'model_config': {
            'num_epochs': 25,
            'fixed_param': False,
            'model_type': 18,
            'version_in_use': storage.VERSION_LATEST
        }
    },
    'AJ#sole': {  # 鞋底
        'model_type': 'resnet',
        'model_config': {
            'num_epochs': 25,
            'fixed_param': False,
            'model_type': 18,
            'version_in_use': storage.VERSION_LATEST
        }
    },
    'AJ#body': {  # 鞋身
        'model_type': 'resnet',
        'model_config': {
            'num_epochs': 25,
            'fixed_param': False,
            'model_type': 18,
            'version_in_use': storage.VERSION_LATEST
        }
    },
    'AJ#inner_body': {  # 鞋内部
        'model_type': 'resnet',
        'model_config': {
            'num_epochs': 25,
            'fixed_param': False,
            'model_type': 18,
            'version_in_use': storage.VERSION_LATEST
        }
    },
    'AJ#tongue': {  # 鞋舌
        'model_type': 'resnet',
        'model_config': {
            'num_epochs': 25,
            'fixed_param': False,
            'model_type': 18,
            'version_in_use': storage.VERSION_LATEST
        }
    },
    'AJ#shoe_tag': {  # 鞋标
        'model_type': 'resnet',
        'model_config': {
            'num_epochs': 25,
            'fixed_param': False,
            'model_type': 18,
            'version_in_use': storage.VERSION_LATEST
        }
    },
    'AJ#shoebox': {  # 鞋盒
        'model_type': 'resnet',
        'model_config': {
            'num_epochs': 25,
            'fixed_param': False,
            'model_type': 18,
            'version_in_use': storage.VERSION_LATEST
        }
    }
}


def get_recognizer(rec_type, model_type=None, model_config=None, use_cache=True):
    if use_cache and rec_type in _recognizer_cache:
        return _recognizer_cache[rec_type]

    if model_config is None:
        model_config = {}

    if model_type:
        child_module = getattr(sys.modules[__name__], model_type)
    else:
        conf = _rec_config.get(rec_type, {})
        model_type = conf.get('model_type')
        if model_type:
            child_module = getattr(sys.modules[__name__], model_type)
            model_config = {**conf.get('model_config', {}), **model_config}
        else:
            raise ModelError('model_type 未指定')

    recognizer = getattr(child_module, 'Recognizer')
    r = recognizer(rec_type, config=model_config)

    if use_cache:
        _recognizer_cache[rec_type] = r

    return r

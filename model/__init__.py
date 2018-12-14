# -*- coding:utf-8 -*-

import sys

from . import resnet, licence_red_stamp, human_face_recognition, ocr_web_image, face_detect, face_match, qrcode
from .error import ModelError
from lib import storage

_recognizer_cache = {}

_rec_config = {
    'AJ_Res': {
        'model_type': 'resnet',
        'model_config': {
            'num_epochs': 50,
            'fixed_param': False,
            'model_type': 50,
            'pretrained_model_name': 'resnet50-19c8e357.pth',
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

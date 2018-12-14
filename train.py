# -*- coding:utf-8 -*-

import argparse

from model import get_recognizer
from lib.logger import get_logger

logger = get_logger('train')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='模型训练脚本')
    parser.add_argument('rec_type', help='识别类别')

    args = parser.parse_args()

    logger.info('---- %s train ----' % args.rec_type)

    recognizer = get_recognizer(args.rec_type, model_config={'train': True}, use_cache=False)
    recognizer.train()

    logger.info('---- %s end ----' % args.rec_type)

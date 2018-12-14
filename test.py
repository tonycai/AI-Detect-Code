# -*- coding:utf-8 -*-

import argparse
import os
import sys
import time
from urllib.request import urlopen

from model import get_recognizer


def test_url(type_, url):
    recognizer = get_recognizer(type_)

    image = urlopen(url).read()
    start_at = time.time()
    result = recognizer.recognize(image)
    print("识别耗时(秒)：", (time.time() - start_at))
    print(result)


def test_file(type_, path):
    recognizer = get_recognizer(type_)

    with open(path, 'rb') as f:
        image = f.read()
        start_at = time.time()
        result = recognizer.recognize(image)
        print("识别耗时(秒)：", (time.time() - start_at))
        print(result)


def test_dir_pure(type_, path):
    recognizer = get_recognizer(type_)
    names = os.listdir(path)
    i = 0
    for name in names:
        i += 1
        sys.stdout.write('%d \r' % i)
        sys.stdout.flush()

        f_path = path + '/' + name
        with open(f_path, 'rb') as f:
            image = f.read()
            result = recognizer.recognize(image)

            print(f_path, result)


def test_dir(type_, path, label_op_threshold):
    label, op, threshold = label_op_threshold.split(' ')
    threshold = float(threshold)

    recognizer = get_recognizer(type_)

    names = os.listdir(path)
    hit_count = 0
    i = 0
    for name in names:
        i += 1
        sys.stdout.write('%d \r' % i)
        sys.stdout.flush()

        f_path = path + '/' + name
        with open(f_path, 'rb') as f:
            image = f.read()
            result = recognizer.recognize(image)

            if op == '>':
                hit = result[label] > threshold
            elif op == '<':
                hit = result[label] < threshold
            else:
                hit = False

            if hit:
                hit_count += 1
            else:
                print(f_path, result[label])

    print('total: %d, hit: %d, rate: %0.2f%%' % (len(names), hit_count, 100.0 * hit_count / len(names)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='模型测试脚本')
    parser.add_argument('-t', dest='type_', help='识别类别')
    parser.add_argument('-p', dest='path', help='图片路径, 支持 url、文件、文件夹')
    parser.add_argument('-lot', dest='lot', help='label op threshold')

    args = parser.parse_args()

    start = time.time()

    if args.path[0:4] == 'http':
        test_url(args.type_, args.path)
    elif os.path.isfile(args.path):
        test_file(args.type_, args.path)
    elif os.path.isdir(args.path):
        if args.lot is None:
            test_dir_pure(args.type_, args.path)
        else:
            test_dir(args.type_, args.path, args.lot)

    print("总耗时(秒)：", (time.time() - start))

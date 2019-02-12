# -*- coding:utf-8 -*-

import argparse
import os
import urllib
import sys
import time
import operator
import random
import string
from urllib.request import urlopen
import cv2

from model import get_recognizer

def classify_file(file_path, new_dir):
    with open(file_path, 'rb') as f:
        image = f.read()
        types = ['AJ#insole', 'AJ#shoe_tag', 'AJ#shoebox', 'AJ#inner_body', 'AJ#body']
        for _type in types:
            new_folder = os.path.abspath(new_dir + '/' + _type)
            if not os.path.exists(new_folder):
                os.makedirs(new_folder)
            recognizer = get_recognizer(_type)
            result = recognizer.recognize(image)
            maxLabel = max(result['labels'].items(), key=operator.itemgetter(1))[0]
            if (maxLabel != 'other'):
                print(_type)
                print(result['labels'][maxLabel])
                file_name = random_char(40) + '.jpg'
                new_file_path = os.sep.join([new_folder, file_name])
                cv2.imwrite(new_file_path,cv2.imread(file_path))
                print(file_path)
                return

def random_char(y):
       return ''.join(random.choice(string.ascii_letters) for x in range(y))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='模型测试脚本')
    parser.add_argument('-p', dest='path', help='图片路径')
    parser.add_argument('-d', dest='dir', help='存储路径')

    args = parser.parse_args()
    for path,dir_list,file_list in os.walk('images/鉴别为真true'):  
        for dir_name in dir_list:
            if(dir_name.find('Jordan') != -1):
                print(os.sep.join([os.path.abspath(path), dir_name]))
                for root, dirs, files in os.walk(os.sep.join([os.path.abspath(path), dir_name])):
                    for image in files:
                        classify_file((os.sep.join([os.path.abspath(root), image])), args.dir)

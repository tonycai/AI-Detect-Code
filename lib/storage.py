# -*- coding:utf-8 -*-

import os
import glob

from .error import BizError

LOCAL_DIR = '/data'

TYPE_MODEL = 'model'
TYPE_SAMPLE = 'sample'

VERSION_LATEST = 'latest'
VERSION_ALL = '*'


def get_local_path(type_, name, version=None):
    path_without_version = '{}/{}/{}'.format(LOCAL_DIR, type_, name)
    if os.path.isdir(path_without_version):
        return path_without_version

    if version == VERSION_LATEST:
        paths = glob.glob('{}#*'.format(path_without_version))
        path_count = len(paths)

        if path_count == 0:
            return None

        # 返回版本号最大的一个
        pv = {}
        for p in paths:
            pv[p] = get_version_from_path(p)

        return max(pv, key=pv.get)

    if version == VERSION_ALL:
        return glob.glob('{}#*'.format(path_without_version))

    if version is not None:
        return '{}#{}'.format(path_without_version, version)

    return path_without_version


def get_version_from_path(path):
    parts = path.rsplit('#', 1)
    return int(parts[1]) if len(parts) == 2 else None


def upload(type_, name, version=None):
    path = get_local_path(type_, name, version)
    if not path:
        raise BizError('数据不存在：{}, {}, {}'.format(type_, name, version))

    if not os.path.isfile(path):
        raise BizError('文件不存在：{}'.format(path))

    # @todo upload to db


def download(type_, name, version=None):
    # @todo get latest version from db if version == VERSION_LATEST

    path = get_local_path(type_, name, version)
    if path and os.path.isfile(path):
        return path

    # @todo download from db

    raise BizError('数据不存在：{}, {}, {}, {}'.format(type_, name, version, path))


def delete_old(type_, name, left_count=3):
    paths = get_local_path(type_, name, VERSION_ALL)
    if len(paths) <= left_count:
        return []

    pvs = [(p, get_version_from_path(p)) for p in paths]
    old_paths = sorted(pvs, key=lambda x: x[1])[0:-left_count]

    deleted_paths = []
    for p, v in old_paths:
        os.remove(p)
        deleted_paths.append(p)

    return deleted_paths

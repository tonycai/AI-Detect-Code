# -*- coding:utf-8 -*-

from urllib.error import URLError
from urllib.request import urlopen

import base64
import datetime

from flask import Flask, request, jsonify

from model import get_recognizer

from lib.logger import get_logger
from lib.error import BizError


app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

logger = get_logger('server')

@app.route('/recognize', methods=['POST'])
def recognize():
    rec_type = request.values.get('type')
    if not rec_type:
        return jsonify({'error_msg': 'type 不能为空'})

    image = request.values.get('image')
    if image:
        image = base64.b64decode(image)
    else:
        image_url = request.values.get('image_url')
        if image_url:
            try:
                logger.info('download image')
                start_time = datetime.datetime.now()
                image = urlopen(image_url, timeout=5).read()
                logger.info('downloaded {:.2f}s'.format(
                    (datetime.datetime.now() - start_time).total_seconds()
                ))
            except URLError as e:
                return jsonify({'error_msg': '[图片下载失败]{}'.format(e)})
        else:
            return jsonify({'error_msg': 'image 或 image_url 不能为空'})

    start_time = datetime.datetime.now()

    try:
        recognizer = get_recognizer(rec_type)
        result = recognizer.recognize(image)
    except BizError as e:
        return jsonify({'error_msg': '[识别错误]{}'.format(e)})

    logger.info('{} {:.2f}s'.format(
        rec_type, (datetime.datetime.now() - start_time).total_seconds()
    ))

    return jsonify(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)

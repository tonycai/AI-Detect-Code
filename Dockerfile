FROM python:3.6.6-stretch

ENV PIP_INDEX_URL=http://pypi.doubanio.com/simple/ PIP_TRUSTED_HOST=pypi.doubanio.com

RUN pip install \
    torch==0.4.0 \
    torchvision==0.2.1 \
    git+https://github.com/pytorch/tnt.git@master \
    opencv-python==3.4.0.12 \
    gunicorn==19.8.1 \
    gevent==1.3.4 \
    Flask==1.0.2 \
    Flask-Cors==3.0.7


RUN apt-get update \
    && apt-get install -y libzbar0 \
    && pip install pyzbar==0.1.7

COPY . /app

WORKDIR /app

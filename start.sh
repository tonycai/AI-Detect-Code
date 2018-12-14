#!/bin/bash

MODE=$1
NAME=smart_identify_server_1
IMAGE_NAME=smart_identify_server

if [ "${MODE}" == "train" ]; then
    echo "nvidia-docker"
    nvidia-docker stop ${NAME}
    nvidia-docker rm ${NAME}
    nvidia-docker run -itd --name ${NAME} -p 3000:3000 --network=smart_identify_default -v /data/smart_identify_data/:/data/ -v /data/smart_identify_server/:/app/ ${IMAGE_NAME} sh -c "gunicorn -w 7 -b 0.0.0.0:3000 -k gevent server:app" --restart=always --env-file docker.prod.env
    else
    docker stop ${NAME}
    docker rm ${NAME}
    docker run -itd --name ${NAME} -p 3000:3000  --network=smart_identify_default  -v /data/smart_identify_data/:/data/ -v /data/smart_identify_server/:/app/ ${IMAGE_NAME} sh -c "gunicorn -w 1 -b 0.0.0.0:3000 -k gevent --reload server:app" --env-file docker.dev.env
fi

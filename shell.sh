#!/bin/bash

MODE=$1
NAME=smart_identify_server_1

if [ "${MODE}" == "train" ]; then
    nvidia-docker exec -it ${NAME} bash
    else
    docker exec -it ${NAME} bash
fi

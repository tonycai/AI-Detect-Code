#!/bin/bash

NAME=smart_identify_server_1

docker logs -f --since 30m ${NAME}

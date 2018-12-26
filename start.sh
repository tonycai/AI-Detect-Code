#!/bin/bash

docker-compose -f docker-compose.prod.yml up --force-recreate --no-deps -d $1

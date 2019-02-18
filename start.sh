#!/bin/bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up --force-recreate --no-deps -d server

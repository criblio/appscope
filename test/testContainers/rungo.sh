#! /bin/bash

sudo service redis-server stop

set -e

docker-compose up go_8
docker-compose down

docker-compose up go_9
docker-compose down

docker-compose up go_10
docker-compose down

docker-compose up go_11
docker-compose down

docker-compose up go_12
docker-compose down

docker-compose up go_13
docker-compose down

docker-compose up go_14
docker-compose down

docker-compose up go_15
docker-compose down

docker-compose up go_16
docker-compose down

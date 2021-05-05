#! /bin/bash

sudo service redis-server stop

set -e

docker-compose up interposed_func
docker-compose down

docker-compose up nginx
docker-compose down

docker-compose up kafka
docker-compose down

docker-compose up elastic
docker-compose down

docker-compose up cribl
docker-compose down

docker-compose up splunk
docker-compose down

docker-compose up detect_proto
docker-compose down

docker-compose up tls
docker-compose down

docker-compose up bash
docker-compose down

docker-compose up gogen
docker-compose down

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

docker-compose up java6 
docker-compose down

docker-compose up java7
docker-compose down

docker-compose up java8 
docker-compose down

docker-compose up java9
docker-compose down

docker-compose up java10 
docker-compose down

docker-compose up java11
docker-compose down

docker-compose up java12
docker-compose down

docker-compose up java13
docker-compose down

docker-compose up java14 
docker-compose down

docker-compose up oracle_java6
docker-compose down

docker-compose up oracle_java7
docker-compose down

docker-compose up oracle_java8
docker-compose down

docker-compose up oracle_java9
docker-compose down

docker-compose up oracle_java10
docker-compose down

docker-compose up oracle_java11
docker-compose down

docker-compose up oracle_java12
docker-compose down

docker-compose up oracle_java13
docker-compose down

docker-compose up oracle_java14
docker-compose down

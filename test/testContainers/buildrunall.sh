#! /bin/bash

set -e

docker-compose build interposed_func
docker-compose up interposed_func
docker-compose down

docker-compose build nginx
docker-compose up nginx
docker-compose down

docker-compose build kafka
docker-compose up kafka
docker-compose down

docker-compose build elastic
docker-compose up elastic
docker-compose down

docker-compose build cribl
docker-compose up cribl
docker-compose down

docker-compose build splunk
docker-compose up splunk
docker-compose down

docker-compose build detect_proto
docker-compose up detect_proto
docker-compose down

docker-compose build tls
docker-compose up tls
docker-compose down

docker-compose build bash
docker-compose up bash
docker-compose down

docker-compose build gogen
docker-compose up gogen
docker-compose down

docker-compose build go_8
docker-compose up go_8
docker-compose down

docker-compose build go_9
docker-compose up go_9
docker-compose down

docker-compose build go_10
docker-compose up go_10
docker-compose down

docker-compose build go_11
docker-compose up go_11
docker-compose down

docker-compose build go_12
docker-compose up go_12
docker-compose down

docker-compose build go_13
docker-compose up go_13
docker-compose down

docker-compose build go_14
docker-compose up go_14
docker-compose down

docker-compose build go_15
docker-compose up go_15
docker-compose down

docker-compose build go_16
docker-compose up go_16
docker-compose down

docker-compose build java6
docker-compose up java6 
docker-compose down

docker-compose build java7
docker-compose up java7
docker-compose down

docker-compose build java8
docker-compose up java8 
docker-compose down

docker-compose build java9
docker-compose up java9
docker-compose down

docker-compose build java10
docker-compose up java10 
docker-compose down

docker-compose build java11
docker-compose up java11
docker-compose down

docker-compose build java12
docker-compose up java12
docker-compose down

docker-compose build java13
docker-compose up java13
docker-compose down

docker-compose build java14
docker-compose up java14 
docker-compose down

docker-compose build oracle_java6
docker-compose up oracle_java6
docker-compose down

docker-compose build oracle_java7
docker-compose up oracle_java7
docker-compose down

docker-compose build oracle_java8
docker-compose up oracle_java8
docker-compose down

docker-compose build oracle_java9
docker-compose up oracle_java9
docker-compose down

docker-compose build oracle_java10
docker-compose up oracle_java10
docker-compose down

docker-compose build oracle_java11
docker-compose up oracle_java11
docker-compose down

docker-compose build oracle_java12
docker-compose up oracle_java12
docker-compose down

docker-compose build oracle_java13
docker-compose up oracle_java13
docker-compose down

docker-compose build oracle_java14
docker-compose up oracle_java14
docker-compose down

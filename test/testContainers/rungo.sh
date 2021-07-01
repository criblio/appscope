#! /bin/bash

set -e

for v in 8 9 10 11 12 13 14 15 16
do 
    docker-compose up go_$v
    docker-compose down
done

#! /bin/bash

set -e

for TEST in $(docker-compose config --services | sort -V); do
    echo 
    echo ======================================================================
    echo Building $TEST Container Image
    echo ======================================================================
    docker-compose build $TEST
    echo 
    echo ======================================================================
    echo Running $TEST Test
    echo ======================================================================
    docker-compose run --rm $TEST
done

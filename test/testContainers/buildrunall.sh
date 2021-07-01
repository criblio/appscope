#! /bin/bash

set -e

for TEST in $(docker-compose config --services | sort -V); do
    echo 
    echo ======================================================================
    echo Building $TEST Container Image
    echo ======================================================================
    time docker-compose build $TEST
    echo 
    echo ======================================================================
    echo Running $TEST Test
    echo ======================================================================
    time docker-compose run --rm $TEST
done

#!/bin/bash
if [ -z $DONT_TIMEOUT ]; then
    echo "Starting timeout script"
    sleep 3720
    echo "killing container on timeout"
    kill -s SIGINT 1
fi
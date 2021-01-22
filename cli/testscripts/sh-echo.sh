#!/bin/sh

idx=1
while true; do
    echo "foo $idx"
    idx=$((idx+1))
    sleep 1
done

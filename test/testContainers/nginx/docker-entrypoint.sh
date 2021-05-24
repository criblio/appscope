#!/bin/bash
set -e

if [ "$1" = "test" ]; then
  netstat -s
  if python3 \
    /opt/test-runner/app.py \
    -v \
    -t nginx \
    -l /opt/test-runner/logs/ \
    -s /usr/lib/libscope.so; then
     exit 0
  else
     netstat -s
     exit 1
  fi
fi

exec "$@"

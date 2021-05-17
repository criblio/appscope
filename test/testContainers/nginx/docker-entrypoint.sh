#!/bin/bash
set -e

if [ "$1" = "test" ]; then
  exec python3 \
    /opt/test-runner/app.py \
    -v \
    -t nginx \
    -l /opt/test-runner/logs/ \
    -s /usr/lib/libscope.so
fi

exec "$@"

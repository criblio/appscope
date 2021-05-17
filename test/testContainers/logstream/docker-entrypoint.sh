#!/bin/bash
set -e

if [ "$1" = "test" ]; then
  exec python3.6 \
    /opt/test-runner/app.py \
    -t cribl \
    -l /opt/test-runner/logs/ \
    -s /usr/lib/libscope.so
fi

exec "$@"

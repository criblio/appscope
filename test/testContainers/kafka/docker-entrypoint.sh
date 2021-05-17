#!/bin/bash
set -e

if [ "$1" = "test" ]; then
  exec /opt/test-runner/bin/python \
    /opt/test-runner/app.py \
    -t kafka \
    -l /opt/test-runner/logs/ \
    -s /usr/lib/libscope.so
fi

exec "$@"

#!/bin/bash
set -e

if [ "$1" = "test" ]; then
  exec /opt/test-runner/bin/python \
    /opt/test-runner/app.py \
    -t syscalls \
    -l /opt/test-runner/logs/ \
    -s /usr/lib/libscope.so \
    --syscalls_tests_config /opt/test-runner/syscall_tests_conf.json
fi

exec "$@"

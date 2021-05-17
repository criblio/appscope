#!/bin/bash
set -e

if [ "$1" = "test" ]; then
  exec /opt/test-runner/bin/test-ssl.sh
fi

exec "$@"

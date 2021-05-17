#!/bin/bash
set -e

if [ "$1" = "test" ]; then
  exec /opt/test-runner/bin/test_tls.sh
fi

exec "$@"

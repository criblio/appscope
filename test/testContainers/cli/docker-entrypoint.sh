#!/bin/bash
set -e

if [ "$1" = "test" ]; then
  exec /opt/test/bin/test_all.sh
fi

exec "$@"

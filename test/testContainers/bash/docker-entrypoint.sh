#!/bin/bash
set -e

if [ "$1" = "test" ]; then
  exec /mybin/test_bash.sh
fi

exec "$@"

#!/bin/bash
set -e

if [ "$1" = "test" ]; then
  exec /go/test_go.sh
fi

exec "$@"

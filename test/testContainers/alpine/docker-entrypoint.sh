#!/bin/bash
set -e

if [ "$1" = "test" ]; then
  echo "Running /go/test_go.sh"
  exec /go/test_go.sh
fi

exec "$@"

#!/bin/bash
set -e

# Start Cribl LogStream
/opt/cribl/bin/cribl start

if [ "$1" = "test" ]; then
  exec /opt/test/bin/scope-test
fi

exec "$@"

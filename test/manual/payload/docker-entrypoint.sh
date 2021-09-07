#!/bin/bash --login

if [ "$1" = "test" ]; then
  exec /usr/local/scope/scope-test
fi

exec "$@"

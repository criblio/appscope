#!/bin/bash

openssl req -newkey rsa:2048 -nodes -keyout /tmp/appscope.key -x509 -days 365 -out /tmp/appscope.crt -subj '/C=US/ST=GA/L=Canton/O=AppScope/OU=IT/CN=appscope'

mkdir /tmp/out
/opt/cribl/bin/cribl start

if [ "$1" = "test" ]; then
  exec /opt/test/bin/scope-test
fi

exec "$@"

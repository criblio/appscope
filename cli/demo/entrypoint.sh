#!/bin/sh

if [ ! -f /etc/nginx/scope-demo.com.crt ]; then
    openssl req -nodes -new -x509 -newkey rsa:2048 -keyout /etc/nginx/scope-demo.com.key -out /etc/nginx/scope-demo.com.crt -days 420 -subj '/O=Scope/C=US/CN=scope-demo.com' >/dev/null 2>/dev/null
    scope nginx
fi

exec "$@"

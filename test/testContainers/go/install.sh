#!/bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

openssl genrsa -out server.key 2048 && \
    openssl ecparam -genkey -name secp384r1 -out server.key && \
    openssl req -new -x509 -sha256 -key server.key -out server.crt \
    -days 3650 -subj "/C=US/ST=California/L=San Francisco/O=Cribl/OU=Cribl/CN=localhost"

source  ~/.gvm/scripts/gvm
for ver in $@; do
    gvm install go$ver -B
done

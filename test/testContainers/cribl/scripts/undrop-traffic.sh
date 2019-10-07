#!/bin/sh
IPTABLES=$(which iptables)
if [ -z ${IPTABLES} ]; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update && apt-get install -y --no-install-recommends iptables
    IPTABLES=$(which iptables)
fi

iptables -D OUTPUT $(iptables -L --line-numbers | grep $1 | awk '{ print $1 }')

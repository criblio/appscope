#!/bin/sh
IPTABLES=$(which iptables)
if [ -z ${IPTABLES} ]; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update && apt-get install -y --no-install-recommends iptables
    IPTABLES=$(which iptables)
fi

iptables -A OUTPUT -p tcp --dport $1 -j DROP

#!/bin/bash

ERR=0

fail() { ERR+=1; echo >&2 "fail:" $@; }

#Check modified configuration
scope service cribl --force --cribldest tls://example_instance.cribl.cloud:10090

if [ ! -f /etc/scope/cribl/scope.yml ]; then
    fail "missing scope.yml"
fi

if [ ! -d /etc/systemd/system/cribl.service.d ]; then
    fail "missing /etc/systemd/system/cribl.service.d/"
fi

if [ ! -f /etc/systemd/system/cribl.service.d/env.conf ]; then
    fail "missing /etc/systemd/system/cribl.service.d/env.conf"
fi

if [ ! -d /var/log/scope ]; then
    fail "missing /var/log/scope/"
fi

if [ ! -d /var/run/scope ]; then
    fail "missing /var/run/scope/"
fi

count=$(grep 'LD_PRELOAD' /etc/systemd/system/cribl.service.d/env.conf | wc -l)
if [ $count -ne 1 ] ; then
    fail "missing LD_PRELOAD in /etc/systemd/system/cribl.service.d/env.conf"
fi

count=$(grep 'SCOPE_HOME' /etc/systemd/system/cribl.service.d/env.conf | wc -l)
if [ $count -ne 1 ] ; then
    fail "missing SCOPE_HOME in /etc/systemd/system/cribl.service.d/env.conf"
fi

count=$(grep 'example_instance' /etc/scope/cribl/scope.yml | wc -l)
if [ $count -ne 1 ] ; then
    fail "Wrong configuration in scope.yml"
fi

#Check default configuration
scope service cribl --force

count=$(grep 'example_instance' /etc/scope/cribl/scope.yml | wc -l)
if [ $count -ne 0 ] ; then
    fail "Wrong configuration in scope.yml"
fi

if [ $ERR -gt 0 ]; then
    echo "$ERR test(s) failed"
    exit $ERR
else
    echo "All test passed"
fi

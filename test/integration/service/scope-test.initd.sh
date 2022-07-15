#!/bin/bash

ERR=0

fail() { ERR+=1; echo >&2 "fail:" $@; }
ARCH=$(uname -m )

CRIBL_ENV_CONF="/etc/sysconfig/cribl"

FULL_CRIBL_CFG="LD_PRELOAD=/usr/lib/${ARCH}-linux-gnu/cribl/libscope.so\nSCOPE_HOME=/etc/scope/cribl"

#Check modified configuration
scope service cribl --force --cribldest tls://example_instance.cribl.cloud:10090 &> /dev/null

if [ ! -f /etc/scope/cribl/scope.yml ]; then
    fail "missing scope.yml"
fi

if [ ! -f $CRIBL_ENV_CONF ]; then
    fail "missing $CRIBL_ENV_CONF"
fi

if [ ! -d /var/log/scope ]; then
    fail "missing /var/log/scope/"
fi

if [ ! -d /var/run/scope ]; then
    fail "missing /var/run/scope/"
fi

count=$(grep 'LD_PRELOAD' $CRIBL_ENV_CONF | wc -l)
if [ $count -ne 1 ] ; then
    fail "missing LD_PRELOAD in $CRIBL_ENV_CONF"
fi

count=$(grep 'SCOPE_HOME' $CRIBL_ENV_CONF | wc -l)
if [ $count -ne 1 ] ; then
    fail "missing SCOPE_HOME in $CRIBL_ENV_CONF"
fi

pcregrep -q -M $FULL_CRIBL_CFG $CRIBL_ENV_CONF
if [ $? -ne "0" ]; then
    fail "missing $FULL_CRIBL_CFG"
    cat $CRIBL_ENV_CONF
fi

count=$(grep 'example_instance' /etc/scope/cribl/scope.yml | wc -l)
if [ $count -ne 1 ] ; then
    fail "Wrong configuration in scope.yml"
fi

#Remove the cribl configuration file
rm $CRIBL_ENV_CONF
touch $CRIBL_ENV_CONF
echo "EXAMPLE_ENV=FOO" >> $CRIBL_ENV_CONF

#Check default configuration
scope service cribl --force &> /dev/null

count=$(grep 'example_instance' /etc/scope/cribl/scope.yml | wc -l)
if [ $count -ne 0 ] ; then
    fail "Wrong configuration in scope.yml"
fi

count=$(grep 'LD_PRELOAD' $CRIBL_ENV_CONF | wc -l)
if [ $count -ne 1 ] ; then
    fail "missing LD_PRELOAD in $CRIBL_ENV_CONF"
fi

count=$(grep 'SCOPE_HOME' $CRIBL_ENV_CONF | wc -l)
if [ $count -ne 1 ] ; then
    fail "missing SCOPE_HOME in $CRIBL_ENV_CONF"
fi

pcregrep -q -M $FULL_CRIBL_CFG $CRIBL_ENV_CONF
if [ $? -ne "0" ]; then
    fail "missing $FULL_CRIBL_CFG"
    cat $CRIBL_ENV_CONF
fi

if [ $ERR -gt 0 ]; then
    echo "$ERR test(s) failed"
    exit $ERR
else
    echo "All test passed"
fi

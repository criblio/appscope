#!/bin/bash

ERR=0

fail() { ERR+=1; echo >&2 "fail:" $@; }

ARCH=$(uname -m )

CRIBL_ENV_CONF="/etc/systemd/system/cribl.service.d/env.conf"
NGINX_ENV_DIR="/etc/systemd/system/nginx.service.d"
NGINX_ENV_CONF="${NGINX_ENV_DIR}/env.conf"

FULL_CRIBL_SERVICE_CFG="[Service].\nEnvironment=LD_PRELOAD=/usr/lib/${ARCH}-linux-gnu/cribl/libscope.so\nEnvironment=SCOPE_HOME=/etc/scope/cribl"

FULL_NGINX_SERVICE_CFG="[Service].\nEnvironment=LD_PRELOAD=/usr/lib/${ARCH}-linux-gnu/cribl/libscope.so\nEnvironment=SCOPE_HOME=/etc/scope/nginx"

#Check modified configuration
scope service cribl --force --cribldest tls://example_instance.cribl.cloud:10090 &> /dev/null

if [ ! -f /etc/scope/cribl/scope.yml ]; then
    fail "missing scope.yml"
fi

if [ ! -d /etc/systemd/system/cribl.service.d ]; then
    fail "missing /etc/systemd/system/cribl.service.d/"
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

pcregrep -q -M $FULL_CRIBL_SERVICE_CFG $CRIBL_ENV_CONF
if [ $? -ne "0" ]; then
    fail "missing $FULL_CRIBL_SERVICE_CFG"
    cat $CRIBL_ENV_CONF
fi

count=$(grep 'example_instance' /etc/scope/cribl/scope.yml | wc -l)
if [ $count -ne 1 ] ; then
    fail "Wrong configuration in scope.yml"
fi

#Check default configuration
scope service cribl --force &> /dev/null

count=$(grep 'example_instance' /etc/scope/cribl/scope.yml | wc -l)
if [ $count -ne 0 ] ; then
    fail "Wrong configuration in scope.yml"
fi

#Create empty configuration file
mkdir -p $NGINX_ENV_DIR
touch $NGINX_ENV_CONF
scope service nginx --force &> /dev/null

count=$(grep 'LD_PRELOAD' $NGINX_ENV_CONF | wc -l)
if [ $count -ne 1 ] ; then
    fail "missing LD_PRELOAD in $NGINX_ENV_CONF"
fi

count=$(grep 'SCOPE_HOME' $NGINX_ENV_CONF | wc -l)
if [ $count -ne 1 ] ; then
    fail "missing SCOPE_HOME in $NGINX_ENV_CONF"
fi

pcregrep -q -M $FULL_NGINX_SERVICE_CFG $NGINX_ENV_CONF
if [ $? -ne "0" ]; then
    fail "missing $FULL_NGINX_SERVICE_CFG"
    cat $NGINX_ENV_CONF
fi

#Create configuration file with Service and Unit section
rm $NGINX_ENV_CONF
touch $NGINX_ENV_CONF
echo "[Service]" >> $NGINX_ENV_CONF
echo "Environment=FOO_BAR=example_env_var" >> $NGINX_ENV_CONF
echo "[Unit]" >> $NGINX_ENV_CONF
echo "Description=Lorem Ipsum" >> $NGINX_ENV_CONF

scope service nginx --force  &> /dev/null

count=$(grep 'LD_PRELOAD' $NGINX_ENV_CONF | wc -l)
if [ $count -ne 1 ] ; then
    fail "missing LD_PRELOAD in $NGINX_ENV_CONF"
fi

count=$(grep 'SCOPE_HOME' $NGINX_ENV_CONF | wc -l)
if [ $count -ne 1 ] ; then
    fail "missing SCOPE_HOME in $NGINX_ENV_CONF"
fi

pcregrep -q -M $FULL_NGINX_SERVICE_CFG $NGINX_ENV_CONF
if [ $? -ne "0" ]; then
    fail "missing $FULL_NGINX_SERVICE_CFG"
    cat $NGINX_ENV_CONF
fi

if [ $ERR -gt 0 ]; then
    echo "$ERR test(s) failed"
    exit $ERR
else
    echo "All test passed"
fi

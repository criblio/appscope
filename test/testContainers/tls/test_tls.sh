#! /bin/bash

declare -i ERR=0
preload=`env | grep LD_PRELOAD`

cd /opt/test

echo "==============================================="
echo "             Testing OpenSSL                   "
echo "==============================================="

./curlssl/src/curl --head https://cribl.io
unset LD_PRELOAD

grep http-req /opt/test-runner/logs/events.log > /dev/null
ERR+=$?

grep "Host: cribl.io" /opt/test-runner/logs/events.log > /dev/null
ERR+=$?

grep http-resp /opt/test-runner/logs/events.log > /dev/null
ERR+=$?

grep HTTP /opt/test-runner/logs/events.log > /dev/null
ERR+=$?

if [ $ERR -eq "0" ]; then
    echo "*************** OpenSSL Success ***************"
else
    echo "*************** OpenSSL Test Failed ***************"
#    cat /opt/test-runner/logs/events.log
fi

rm /opt/test-runner/logs/events.log

echo "==============================================="
echo "             Testing gnutls                    "
echo "==============================================="
export $preload
./curltls/src/curl --head https://cribl.io
unset LD_PRELOAD

grep http-req /opt/test-runner/logs/events.log > /dev/null
ERR+=$?

grep "Host: cribl.io" /opt/test-runner/logs/events.log > /dev/null
ERR+=$?

grep http-resp /opt/test-runner/logs/events.log > /dev/null
ERR+=$?

grep HTTP /opt/test-runner/logs/events.log > /dev/null
ERR+=$?

if [ $ERR -eq "0" ]; then
    echo "*************** gnutls Success ***************"
else
    echo "*************** gnutls Test Failed ***************"
#    cat /opt/test-runner/logs/events.log
fi

rm /opt/test-runner/logs/events.log

echo "==============================================="
echo "             Testing nss                       "
echo "==============================================="
export $preload
curl --head https://cribl.io
unset LD_PRELOAD

grep http-req /opt/test-runner/logs/events.log > /dev/null
ERR+=$?

grep "Host: cribl.io" /opt/test-runner/logs/events.log > /dev/null
ERR+=$?

grep http-resp /opt/test-runner/logs/events.log > /dev/null
ERR+=$?

grep HTTP /opt/test-runner/logs/events.log > /dev/null
ERR+=$?

if [ $ERR -eq "0" ]; then
    echo "*************** nss Success ***************"
else
    echo "*************** nss Test Failed ***************"
#    cat /opt/test-runner/logs/events.log
fi

rm /opt/test-runner/logs/events.log

echo "==============================================="
echo "      Testing hot patch with node.js           "
echo "==============================================="
export $preload
echo "Running node wih an HTTPS request"
node /opt/test-runner/bin/nodehttp.ts > /dev/null
unset LD_PRELOAD

grep http-req /opt/test-runner/logs/events.log > /dev/null
ERR+=$?

grep "Host: cribl.io" /opt/test-runner/logs/events.log > /dev/null
ERR+=$?

grep http-resp /opt/test-runner/logs/events.log > /dev/null
ERR+=$?

grep HTTP /opt/test-runner/logs/events.log > /dev/null
ERR+=$?

if [ $ERR -eq "0" ]; then
    echo "*************** hot patch Success ***************"
else
    echo "*************** host patch Test Failed ***************"
#    cat /opt/test-runner/logs/events.log
fi

rm /opt/test-runner/logs/events.log

if [ $ERR -eq "0" ]; then

    echo "*************** Test Passed ***************"
else
    echo "*************** Test Failed ***************"
fi

echo ""
echo "==============================================="
echo "      Testing Python with HTTPS                "
echo "==============================================="
/opt/rh/rh-python36/root/usr/bin/pip3.6 install pyopenssl
export $preload
echo "Running Python wih an HTTPS request"
/opt/rh/rh-python36/root/usr/bin/python3.6 /opt/test-runner/bin/testssl.py create_certs
/opt/rh/rh-python36/root/usr/bin/python3.6 /opt/test-runner/bin/testssl.py start_server&
/opt/rh/rh-python36/root/usr/bin/python3.6 /opt/test-runner/bin/testssl.py run_client

grep http-req /opt/test-runner/logs/events.log > /dev/null
ERR+=$?

grep "Host: cribl.io" /opt/test-runner/logs/events.log > /dev/null
ERR+=$?

grep http-resp /opt/test-runner/logs/events.log > /dev/null
ERR+=$?

grep HTTP /opt/test-runner/logs/events.log > /dev/null
ERR+=$?

if [ $ERR -eq "0" ]; then
    echo "*************** Python Success ***************"
else
    echo "*************** Python Test Failed ***************"
#    cat /opt/test-runner/logs/events.log
fi

rm /opt/test-runner/logs/events.log

exit ${ERR}

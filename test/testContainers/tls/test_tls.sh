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

echo "==============================================="
echo "             Testing Ruby                      "
echo "==============================================="

echo "Creating key files for ruby client and server"
(cd /opt/test-runner/ruby && openssl req -x509 -newkey rsa:4096 -keyout priv.pem -out cert.pem -days 365 -nodes -subj "/C=US/ST=California/L=San Francisco/O=Cribl/OU=Cribl/CN=localhost")
export $preload
echo "Starting to test ruby"
RUBY_HTTP_START=$(grep http- /opt/test-runner/logs/events.log | grep -c 10101)
(cd /opt/test-runner/ruby && ./server.rb 10101 &)
sleep 1
(cd /opt/test-runner/ruby && ./client.rb 127.0.0.1 10101)
sleep 1
echo "Done testing ruby"
RUBY_HTTP_END=$(grep http- /opt/test-runner/logs/events.log | grep -c 10101)

if (( $RUBY_HTTP_END - $RUBY_HTTP_START >= 6 )); then
    echo "*************** Ruby Passed ***************"
else
    echo "*************** Ruby Failed ***************"
    echo RUBY_HTTP_START=$RUBY_HTTP_START
    echo RUBY_HTTP_END=$RUBY_HTTP_END
    ERR+=1
fi

rm /opt/test-runner/logs/events.log

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

echo ""
echo "==============================================="
echo "      Testing Rust app with HTTPS                "
echo "==============================================="
export $preload
echo "Running a Rust app with an HTTPS request"
/opt/test-runner/bin/http_test > /dev/null

grep http-req /opt/test-runner/logs/events.log > /dev/null
ERR+=$?

grep "Host: cribl.io" /opt/test-runner/logs/events.log > /dev/null
ERR+=$?

grep http-resp /opt/test-runner/logs/events.log > /dev/null
ERR+=$?

grep HTTP /opt/test-runner/logs/events.log > /dev/null
ERR+=$?

if [ $ERR -eq "0" ]; then
    echo "*************** Rust Success ***************"
else
    echo "*************** Rust Test Failed ***************"
#    cat /opt/test-runner/logs/events.log
fi

rm /opt/test-runner/logs/events.log


echo ""
echo "==============================================="
echo "      Testing php with HTTPS                   "
echo "==============================================="
export $preload
echo "Starting to test php"
PHP_HTTP_START=$(grep http- /opt/test-runner/logs/events.log | grep -c sslclient.php)
php /opt/test-runner/php/sslclient.php > /dev/null
echo "Done testing php"
PHP_HTTP_END=$(grep http- /opt/test-runner/logs/events.log | grep -c sslclient.php)

# We really expect *3*, not *2*.  We're missing http-req right now.
if (( $PHP_HTTP_END - $PHP_HTTP_START >= 2 )); then
    echo "*************** php Passed ***************"
else
    echo "*************** php Failed ***************"
    echo PHP_HTTP_START=$PHP_HTTP_START
    echo PHP_HTTP_END=$PHP_HTTP_END
    ERR+=1
fi


exit ${ERR}

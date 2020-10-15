#! /bin/bash

DEBUG=0  # set this to 1 to capture the EVT_FILE for each test

FAILED_TEST_LIST=""
FAILED_TEST_COUNT=0

PRELOAD=`env | grep LD_PRELOAD`
EVT_FILE="/opt/test-runner/logs/events.log"

starttest(){
    CURRENT_TEST=$1
    echo "==============================================="
    echo "             Testing $CURRENT_TEST             "
    echo "==============================================="
    export $PRELOAD
    ERR=0
}

evaltest(){
    echo "             Evaluating $CURRENT_TEST"
    unset LD_PRELOAD
}

endtest(){
    if [ $ERR -eq "0" ]; then
        RESULT=PASSED
    else
        RESULT=FAILED
        FAILED_TEST_LIST+=$CURRENT_TEST
        FAILED_TEST_LIST+=" "
        FAILED_TEST_COUNT=$(($FAILED_TEST_COUNT + 1))
    fi

    echo "*************** $CURRENT_TEST $RESULT ***************"
    echo ""
    echo ""

    # copy the EVT_FILE to help with debugging
    if (( $DEBUG )) || [ $RESULT == "FAILED" ]; then
        cp $EVT_FILE $EVT_FILE.$CURRENT_TEST
    fi

    rm $EVT_FILE
}


#
# OpenSSL
#
starttest OpenSSL
cd /opt/test
./curlssl/src/curl --head https://cribl.io
evaltest

grep http-req $EVT_FILE > /dev/null
ERR+=$?

grep '"http.host":"cribl.io"' $EVT_FILE > /dev/null
ERR+=$?

grep http-resp $EVT_FILE > /dev/null
ERR+=$?

grep HTTP $EVT_FILE > /dev/null
ERR+=$?
endtest


#
# gnutls
#
starttest gnutls
./curltls/src/curl --head https://cribl.io
evaltest

grep http-req $EVT_FILE > /dev/null
ERR+=$?

grep '"http.host":"cribl.io"' $EVT_FILE > /dev/null
ERR+=$?

grep http-resp $EVT_FILE > /dev/null
ERR+=$?

grep HTTP $EVT_FILE > /dev/null
ERR+=$?
endtest


#
# nss
#
starttest nss
curl --head https://cribl.io
evaltest

grep http-req $EVT_FILE > /dev/null
ERR+=$?

grep '"http.host":"cribl.io"' $EVT_FILE > /dev/null
ERR+=$?

grep http-resp $EVT_FILE > /dev/null
ERR+=$?

grep HTTP $EVT_FILE > /dev/null
ERR+=$?
endtest


#
# node.js
#
starttest "node.js"
node /opt/test-runner/bin/nodehttp.ts > /dev/null
evaltest

grep http-req $EVT_FILE > /dev/null
ERR+=$?

grep '"http.host":"cribl.io"' $EVT_FILE > /dev/null
ERR+=$?

grep http-resp $EVT_FILE > /dev/null
ERR+=$?

grep HTTP $EVT_FILE > /dev/null
ERR+=$?
endtest


#
# Ruby
#
echo "Creating key files for ruby client and server"
(cd /opt/test-runner/ruby && openssl req -x509 -newkey rsa:4096 -keyout priv.pem -out cert.pem -days 365 -nodes -subj "/C=US/ST=California/L=San Francisco/O=Cribl/OU=Cribl/CN=localhost")
starttest Ruby
RUBY_HTTP_START=$(grep http- $EVT_FILE | grep -c 10101)
(cd /opt/test-runner/ruby && ./server.rb 10101 &)
sleep 1
(cd /opt/test-runner/ruby && ./client.rb 127.0.0.1 10101)
sleep 1
evaltest
RUBY_HTTP_END=$(grep http- $EVT_FILE | grep -c 10101)

if (( $RUBY_HTTP_END - $RUBY_HTTP_START < 6 )); then
    ERR+=1
fi
endtest


#
# Python
#
/opt/rh/rh-python36/root/usr/bin/pip3.6 install pyopenssl
starttest Python
/opt/rh/rh-python36/root/usr/bin/python3.6 /opt/test-runner/bin/testssl.py create_certs
/opt/rh/rh-python36/root/usr/bin/python3.6 /opt/test-runner/bin/testssl.py start_server&
sleep 1
/opt/rh/rh-python36/root/usr/bin/python3.6 /opt/test-runner/bin/testssl.py run_client
sleep 1
evaltest

COUNT=$(grep -c http- $EVT_FILE)
if (( $COUNT < 6 )); then
    ERR+=1
fi
endtest


#
# Rust
#
starttest Rust
/opt/test-runner/bin/http_test > /dev/null
evaltest

grep http-req $EVT_FILE > /dev/null
ERR+=$?

grep '"http.host":"cribl.io"' $EVT_FILE > /dev/null
ERR+=$?

grep http-resp $EVT_FILE > /dev/null
ERR+=$?

grep HTTP $EVT_FILE > /dev/null
ERR+=$?
endtest


#
# php
#
starttest php
PHP_HTTP_START=$(grep http- $EVT_FILE | grep -c sslclient.php)
php /opt/test-runner/php/sslclient.php > /dev/null
evaltest

PHP_HTTP_END=$(grep http- $EVT_FILE | grep -c sslclient.php)

if (( $PHP_HTTP_END - $PHP_HTTP_START < 3 )); then
    ERR+=1
fi
endtest


#
# apache
#
starttest apache
APACHE_HTTP_START=$(grep http- $EVT_FILE | grep -c httpd)
httpd -k start
curl -k https://localhost:443/ > /dev/null
httpd -k stop
evaltest
APACHE_HTTP_END=$(grep http- $EVT_FILE | grep -c httpd)

if (( $APACHE_HTTP_END - $APACHE_HTTP_START < 3 )); then
    ERR+=1
fi
endtest


if (( $FAILED_TEST_COUNT == 0 )); then
    echo ""
    echo ""
    echo "*************** ALL TESTS PASSED ***************"
else
    echo "*************** SOME TESTS FAILED ***************"
    echo "Failed tests: $FAILED_TEST_LIST"
    echo "Refer to these files for more info:"
    for FAILED_TEST in $FAILED_TEST_LIST; do
        echo "  $EVT_FILE.$FAILED_TEST"
    done
fi

exit ${FAILED_TEST_COUNT}

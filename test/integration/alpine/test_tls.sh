#! /bin/bash

DEBUG=0  # set this to 1 to capture the EVT_FILE for each test

FAILED_TEST_LIST=""
FAILED_TEST_COUNT=0

EVT_FILE="/opt/test/logs/events.log"

starttest(){
    CURRENT_TEST=$1
    echo "==============================================="
    echo "             Testing $CURRENT_TEST             "
    echo "==============================================="
    ERR=0
}

evaltest(){
    echo "             Evaluating $CURRENT_TEST"
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
    touch $EVT_FILE
}

export SCOPE_PAYLOAD_ENABLE=true
export SCOPE_PAYLOAD_HEADER=true

evalPayload(){
    PAYLOADERR=0
    echo "Testing that payload files don't contain tls for $CURRENT_TEST"
    for FILE in $(ls /tmp/*in /tmp/*out 2>/dev/null); do
        # Continue if there aren't any .in or .out files
        if [ $? -ne "0" ]; then
            continue
        fi

        hexdump -C $FILE | cut -c11-58 | \
                     egrep "7d[ \n]+0a[ \n]+1[4-7][ \n]+03[ \n]+0[0-3]"
        if [ $? -eq "0" ]; then
            echo "$FILE contains tls"
            PAYLOADERR=$(($PAYLOADERR + 1))
        fi
    done

    # There were failures.  Move them out of the way before continuing.
    if [ $PAYLOADERR -ne "0" ]; then
        echo "Moving payload files to /tmp/payload/$CURRENT_TEST"
        mkdir -p /tmp/payload/$CURRENT_TEST
        cp /tmp/*in /tmp/payload/$CURRENT_TEST
        cp /tmp/*out /tmp/payload/$CURRENT_TEST
        rm /tmp/*in /tmp/*out
    fi

    return $PAYLOADERR
}


#
# OpenSSL
#
starttest OpenSSL
cd /opt/test
scope -z /opt/test/curl-ssl --http1.1 --head https://cribl.io
evaltest

grep http.req $EVT_FILE > /dev/null
ERR+=$?

grep http.resp $EVT_FILE > /dev/null
ERR+=$?

grep dns.req $EVT_FILE > /dev/null
ERR+=$?

grep dns.resp $EVT_FILE > /dev/null
ERR+=$?

cat $EVT_FILE

evalPayload
ERR+=$?

endtest


#
# gnutls
#
starttest gnutls
scope -z /opt/test/curl-tls --http1.1 --head https://cribl.io
evaltest

grep http.req $EVT_FILE > /dev/null
ERR+=$?

grep http.resp $EVT_FILE > /dev/null
ERR+=$?

grep dns.req $EVT_FILE > /dev/null
ERR+=$?

grep dns.resp $EVT_FILE > /dev/null
ERR+=$?

evalPayload
ERR+=$?

endtest


#
# nss
#
starttest nss
scope -z /opt/test/curl-nss --http1.1 --head https://cribl.io
evaltest

grep http.req $EVT_FILE > /dev/null
ERR+=$?

grep http.resp $EVT_FILE > /dev/null
ERR+=$?

grep dns.req $EVT_FILE > /dev/null
ERR+=$?

grep dns.resp $EVT_FILE > /dev/null
ERR+=$?

evalPayload
ERR+=$?

endtest


#
# node.js
#
starttest "node.js"
scope -z node /opt/test/bin/nodehttp.ts > /dev/null
evaltest

grep http.req $EVT_FILE > /dev/null
ERR+=$?

grep http.resp $EVT_FILE > /dev/null
ERR+=$?

evalPayload
ERR+=$?

endtest


#
# Ruby
#
starttest Ruby
echo "Creating key files for ruby client and server"
(cd /opt/test/bin && openssl req -x509 -newkey rsa:4096 -keyout priv.pem -out cert.pem -days 365 -nodes -subj "/C=US/ST=California/L=San Francisco/O=Cribl/OU=Cribl/CN=localhost")
RUBY_HTTP_START=$(grep "http\." $EVT_FILE | grep 10101 | wc -l)
(cd /opt/test/bin && scope -z ./server.rb 10101 &)
sleep 1
(cd /opt/test/bin && scope -z ./client.rb 127.0.0.1 10101)
sleep 1
evaltest
RUBY_HTTP_END=$(grep "http\." $EVT_FILE | grep 10101 | wc -l)

if (( $RUBY_HTTP_END - $RUBY_HTTP_START < 4 )); then
    ERR+=1
fi

evalPayload
ERR+=$?

endtest


#
# Python
#
starttest Python
scope -z python3 /opt/test/bin/testssl.py create_certs
scope -z python3 /opt/test/bin/testssl.py start_server&
sleep 1
scope -z python3 /opt/test/bin/testssl.py run_client
sleep 1
evaltest

COUNT=$(grep "http\." $EVT_FILE | wc -l)
if (( $COUNT < 4 )); then
    ERR+=1
fi

evalPayload
ERR+=$?

endtest


#
# Rust
#
starttest Rust
scope -z /opt/test/bin/http_test > /dev/null
evaltest

grep http.req $EVT_FILE > /dev/null
ERR+=$?

grep http.resp $EVT_FILE > /dev/null
ERR+=$?

evalPayload
ERR+=$?

endtest


#
# php
#
starttest php
PHP_HTTP_START=$(grep "http\." $EVT_FILE | grep sslclient.php | wc -l)
scope -z php /opt/test/bin/sslclient.php > /dev/null
evaltest

PHP_HTTP_END=$(grep "http\." $EVT_FILE | grep sslclient.php | wc -l)

if (( $PHP_HTTP_END - $PHP_HTTP_START < 2 )); then
    ERR+=1
fi

evalPayload
ERR+=$?

endtest


#
# apache
#
starttest apache
APACHE_HTTP_START=$(grep "http\." $EVT_FILE | grep httpd | wc -l)
scope -z httpd -k start
scope -z curl -k https://localhost:443/
scope -z httpd -k stop
evaltest
APACHE_HTTP_END=$(grep "http\." $EVT_FILE | grep httpd | wc -l)

if (( $APACHE_HTTP_END - $APACHE_HTTP_START < 2 )); then
    ERR+=1
fi

evalPayload
ERR+=$?

endtest

unset SCOPE_PAYLOAD_ENABLE
unset SCOPE_PAYLOAD_HEADER

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

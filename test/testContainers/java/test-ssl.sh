#! /bin/bash

# in some cases, the /etc/profile.d isn't loaded so force this
export PATH="/usr/local/scope:/usr/local/scope/bin:${PATH}"

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

export SCOPE_PAYLOAD_ENABLE=true
export SCOPE_PAYLOAD_HEADER=true

evalPayload(){
    PAYLOADERR=0
    if ! command -v hexdump; then
        echo "hexdump is not available; skipping test of tls in payload files for $CURRENT_TEST"
        return $PAYLOADERR
    fi

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



starttest Tomcat
ldscope /opt/tomcat/bin/catalina.sh run &
evaltest

until [ "`curl $CURL_PARAMS  -k --silent --connect-timeout 1 -I https://localhost:8443 | grep 'Coyote'`" != "" ];
do
    echo waiting for tomcat...
    sleep 1
done

sleep 2
grep http-req $EVT_FILE > /dev/null
ERR+=$?

grep http-resp $EVT_FILE > /dev/null
ERR+=$?

grep '"net_peer_ip":"127.0.0.1"' $EVT_FILE > /dev/null
ERR+=$?

grep -E '"net_peer_port":"[0-9]+"' $EVT_FILE > /dev/null
ERR+=$?

evalPayload
ERR+=$?

endtest


starttest SSLSocketClient
cd /opt/javassl
ldscope java -Djavax.net.ssl.trustStore=/opt/tomcat/certs/tomcat.p12 -Djavax.net.ssl.trustStorePassword=changeit -Djavax.net.ssl.trustStoreType=pkcs12 SSLSocketClient > /dev/null
evaltest
grep http-req $EVT_FILE > /dev/null
ERR+=$?

grep http-resp $EVT_FILE > /dev/null
ERR+=$?

grep '"net_peer_ip":"127.0.0.1"' $EVT_FILE > /dev/null
ERR+=$?

grep -E '"net_peer_port":"[0-9]+"' $EVT_FILE > /dev/null
ERR+=$?

evalPayload
ERR+=$?

endtest

/opt/tomcat/bin/catalina.sh stop
sleep 3


if [ "x86_64" = "$(uname -m)" ]; then # x86_64 only
#
# Java HTTP Server
#


starttest java_http_attach_curl

cd /opt/java_http
java SimpleHttpServer 2> /dev/null &
HTTP_SERVER_PID=$!
sleep 1
evaltest
ldscope --attach ${HTTP_SERVER_PID}
curl http://localhost:8000/status
sleep 5

grep -q '"proc":"java"' $EVT_FILE > /dev/null
ERR+=$?

grep -q http-req $EVT_FILE > /dev/null
ERR+=$?

grep -q http-resp $EVT_FILE > /dev/null
ERR+=$?

grep -q fs.open $EVT_FILE > /dev/null
ERR+=$?

grep -q fs.close $EVT_FILE > /dev/null
ERR+=$?

grep -q net.conn.open $EVT_FILE > /dev/null
ERR+=$?

grep -q net.conn.close $EVT_FILE > /dev/null
ERR+=$?

kill -9 ${HTTP_SERVER_PID}

endtest

starttest java_http_curl_attach_curl

cd /opt/java_http
java SimpleHttpServer 2> /dev/null &
HTTP_SERVER_PID=$!
sleep 1
evaltest
curl http://localhost:8000/status
ldscope --attach ${HTTP_SERVER_PID}
curl http://localhost:8000/status
sleep 5

grep -q '"proc":"java"' $EVT_FILE > /dev/null
ERR+=$?

grep -q http-req $EVT_FILE > /dev/null
ERR+=$?

grep -q http-resp $EVT_FILE > /dev/null
ERR+=$?

grep -q fs.open $EVT_FILE > /dev/null
ERR+=$?

grep -q fs.close $EVT_FILE > /dev/null
ERR+=$?

grep -q net.conn.open $EVT_FILE > /dev/null
ERR+=$?

grep -q net.conn.close $EVT_FILE > /dev/null
ERR+=$?

kill -9 ${HTTP_SERVER_PID}

endtest

starttest java_https_attach_curl
/opt/tomcat/bin/catalina.sh run &
TOMCAT_PID=$!
sleep 3
evaltest

ldscope --attach ${TOMCAT_PID}

curl -k https://localhost:8443
sleep 5

grep http-req $EVT_FILE > /dev/null
ERR+=$?

grep http-resp $EVT_FILE > /dev/null
ERR+=$?

evalPayload
ERR+=$?

/opt/tomcat/bin/catalina.sh stop
sleep 3

endtest
# TODO uncomment this with full support for attach
# when attach will work for already loaded libraries

# starttest java_https_curl_attach_curl
# /opt/tomcat/bin/catalina.sh run &
# TOMCAT_PID=$!
# sleep 3
# curl -k https://localhost:8443
# sleep 5
# evaltest

# ldscope --attach ${TOMCAT_PID}

# curl -k https://localhost:8443
# sleep 5

# grep http-req $EVT_FILE > /dev/null
# ERR+=$?

# grep http-resp $EVT_FILE > /dev/null
# ERR+=$?

# evalPayload
# ERR+=$?

# /opt/tomcat/bin/catalina.sh stop
# sleep 3

# endtest

# TODO: Java9 fails see issue #630
# remove if condition below after fixing the issue
if [[ -z "${SKIP_LDSCOPE_TEST}" ]]; then
starttest java_http_ldscope

cd /opt/java_http
ldscope java SimpleHttpServer 2> /dev/null &
HTTP_SERVER_PID=$!
evaltest
sleep 1
curl http://localhost:8000/status
sleep 5

grep -q '"proc":"java"' $EVT_FILE > /dev/null
ERR+=$?

grep -q http-req $EVT_FILE > /dev/null
ERR+=$?

grep -q http-resp $EVT_FILE > /dev/null
ERR+=$?

grep -q fs.open $EVT_FILE > /dev/null
ERR+=$?

grep -q fs.close $EVT_FILE > /dev/null
ERR+=$?

grep -q net.conn.open $EVT_FILE > /dev/null
ERR+=$?

grep -q net.conn.close $EVT_FILE > /dev/null
ERR+=$?

kill -9 ${HTTP_SERVER_PID}
sleep 1

endtest

fi

fi # x86_64 only

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


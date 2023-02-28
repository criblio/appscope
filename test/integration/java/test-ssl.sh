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

evalInternalEvents(){
    echo "Testing that eventsdon't contain internal events for $CURRENT_TEST"
    # verify that we don't scope event from ourselves
    # periodic -> mtcConnect -> transportConnect -> socketConnectionStart -> getAddressList
    count=$(grep -E '"source":"fs.open".*"file":"/etc/hosts"' $EVT_FILE | wc -l)
    if [ $count -ne 0 ] ; then
        ERR+=1
    fi

    count=$(grep -E '"source":"fs.close".*"file":"/etc/hosts"' $EVT_FILE | wc -l)
    if [ $count -ne 0 ] ; then
        ERR+=1
    fi
}

starttest Tomcat
scope -z /opt/tomcat/bin/catalina.sh run &
evaltest

CURL_MAX_RETRY=10
until [[ "`curl $CURL_PARAMS -k --silent --connect-timeout 1 -I https://localhost:8443 | grep 'Coyote'`" != "" ]] || [[ "$CURL_MAX_RETRY" -lt 0 ]];
do
    echo waiting for tomcat...
    sleep 1
    let CURL_MAX_RETRY-=1
done

if [[ "$CURL_MAX_RETRY" -lt 0 ]]; then
    echo "Error: timed out waiting for tomcat"
    ERR+=$?
fi

sleep 2
grep http.req $EVT_FILE > /dev/null
ERR+=$?

grep http.resp $EVT_FILE > /dev/null
ERR+=$?

grep '"net_peer_ip":"127.0.0.1"' $EVT_FILE > /dev/null
ERR+=$?

grep '"net_peer_port":' $EVT_FILE > /dev/null
ERR+=$?

evalPayload
ERR+=$?

endtest


starttest SSLSocketClient
cd /opt/javassl
scope -z java -Djavax.net.ssl.trustStore=/opt/tomcat/certs/tomcat.p12 -Djavax.net.ssl.trustStorePassword=changeit -Djavax.net.ssl.trustStoreType=pkcs12 SSLSocketClient > /dev/null
evaltest
grep http.req $EVT_FILE > /dev/null
ERR+=$?

grep http.resp $EVT_FILE > /dev/null
ERR+=$?

grep '"net_peer_ip":"127.0.0.1"' $EVT_FILE > /dev/null
ERR+=$?

grep -E '"net_peer_port":' $EVT_FILE > /dev/null
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
scope --ldattach ${HTTP_SERVER_PID}
curl http://localhost:8000/status
sleep 5

grep -q '"proc":"java"' $EVT_FILE > /dev/null
ERR+=$?

grep -q http.req $EVT_FILE > /dev/null
ERR+=$?

grep -q http.resp $EVT_FILE > /dev/null
ERR+=$?

grep -q fs.open $EVT_FILE > /dev/null
ERR+=$?

grep -q fs.close $EVT_FILE > /dev/null
ERR+=$?

grep -q net.open $EVT_FILE > /dev/null
ERR+=$?

grep -q net.close $EVT_FILE > /dev/null
ERR+=$?

evalInternalEvents

kill -9 ${HTTP_SERVER_PID}

endtest

starttest java_http_curl_attach_curl

cd /opt/java_http
java SimpleHttpServer 2> /dev/null &
HTTP_SERVER_PID=$!
sleep 1
evaltest
curl http://localhost:8000/status
scope --ldattach ${HTTP_SERVER_PID}
curl http://localhost:8000/status
sleep 5

grep -q '"proc":"java"' $EVT_FILE > /dev/null
ERR+=$?

grep -q http.req $EVT_FILE > /dev/null
ERR+=$?

grep -q http.resp $EVT_FILE > /dev/null
ERR+=$?

grep -q net.open $EVT_FILE > /dev/null
ERR+=$?

grep -q net.close $EVT_FILE > /dev/null
ERR+=$?

evalInternalEvents

kill -9 ${HTTP_SERVER_PID}

endtest

# TODO: Java9 fails see issue #630
# remove if condition below after fixing the issue
if [[ -z "${SKIP_SCOPE_TEST}" ]]; then
starttest java_http_scope

cd /opt/java_http
scope -z java SimpleHttpServer 2> /dev/null &
HTTP_SERVER_PID=$!
evaltest
sleep 1
curl http://localhost:8000/status
sleep 5

grep -q '"proc":"java"' $EVT_FILE > /dev/null
ERR+=$?

grep -q http.req $EVT_FILE > /dev/null
ERR+=$?

grep -q http.resp $EVT_FILE > /dev/null
ERR+=$?

grep -q fs.open $EVT_FILE > /dev/null
ERR+=$?

grep -q fs.close $EVT_FILE > /dev/null
ERR+=$?

grep -q net.open $EVT_FILE > /dev/null
ERR+=$?

grep -q net.close $EVT_FILE > /dev/null
ERR+=$?

evalInternalEvents

kill -9 ${HTTP_SERVER_PID}
sleep 1

endtest

fi

fi # x86_64 only


#starttest java_crash_analysis
#
#cd /opt/java_http
#
## if we crash java, verify that we capture a backtrace & coredump
#scope run -p --backtrace --coredump -- java SimpleHttpServer 2> /dev/null &
#HTTP_SERVER_PID=$!
#sleep 1
#
#evaltest
#
## we'll send a signal to act like a java crash
#kill -s SIGBUS $HTTP_SERVER_PID
#sleep 2
#
#grep -q '"proc":"java"' $EVT_FILE > /dev/null
#ERR+=$?
#
## test that the expected files have been produced
#snapshot_dir="/tmp/appscope/$HTTP_SERVER_PID"
#ls $snapshot_dir/info* 1>/dev/null
#ERR+=$?
#ls $snapshot_dir/cfg* 1>/dev/null
#ERR+=$?
#ls $snapshot_dir/backtrace* 1>/dev/null
#ERR+=$?
#ls $snapshot_dir/core* 1>/dev/null
#ERR+=$?
#
#sleep 5
## test that the SimpleHttpServer has been terminated by the SIGBUS.
## kill -0 allows us to check if the pid is still running.
#if kill -0 $HTTP_SERVER_PID; then
#    ERR+=1
#fi
#
## if it is still running, kill it
#kill -9 $HTTP_SERVER_PID
#
#endtest
#
#
#ls -al $snapshot_dir

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


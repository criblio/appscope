#! /bin/bash

DEBUG=0 # set this to 1 to capture the EVT_FILE for each test
FAILED_TEST_LIST=""
FAILED_TEST_COUNT=0
SCOPE_LOG_DEST=file:///tmp/scope.log
SCOPE_EVENT_DEST=file:///tmp/events.log
SCOPE_METRIC_DEST=file:///tmp/events.log
SCOPE_CRIBL_ENABLE=false
SCOPE_LOG_LEVEL=error
SCOPE_METRIC_VERBOSITY=4
SCOPE_EVENT_LOGFILE=true
SCOPE_EVENT_CONSOLE=true
SCOPE_EVENT_METRIC=true
SCOPE_EVENT_HTTP=true
EVT_FILE="/tmp/events.json"
MTC_FILE="/tmp/metrics.json"
LOG_FILE="/tmp/scope.log"

starttest(){
    CURRENT_TEST=$1
    echo "=============================================="
    echo "             Testing $CURRENT_TEST            "
    echo "=============================================="
    ERR=0

    touch $EVT_FILE
    touch $MTC_FILE
    touch $LOG_FILE
}

endtest(){
    if [ $ERR -eq "0" ]; then
        RESULT=PASSED
    else
        cat $LOG_FILE
        cat $EVT_FILE
        cat $MTC_FILE
        RESULT=FAILED
        FAILED_TEST_LIST+=$CURRENT_TEST
        FAILED_TEST_LIST+=" "
        FAILED_TEST_COUNT=$(($FAILED_TEST_COUNT + 1))
    fi

    echo "******************* $RESULT *******************"
    echo ""
    echo ""
    
    # copy the EVT_FILE to help with debugging
    if (( $DEBUG )) || [ $RESULT == "FAILED" ]; then
        cp $EVT_FILE $EVT_FILE.$CURRENT_TEST
        cp $MTC_FILE $MTC_FILE.$CURRENT_TEST
        cp $LOG_FILE $LOG_FILE.$CURRENT_TEST
    fi

    rm -f $EVT_FILE
    rm -f $MTC_FILE
    rm -f $LOG_FILE
}


################# TESTS ################# 

# Init test
starttest "nginx http1"

# Run scoped nginx
scope run -- nginx
sleep 2

# Make a http request with curl, to the nginx server
curl --http1.1 -k http://localhost/
sleep 2

# Cleanly exit nginx, allowing metrics and events to be flushed
nginx -s stop
sleep 1

# Validate that expected events were produced
grep net.app $EVT_FILE > /dev/null
ERR+=$?
grep net.open $EVT_FILE > /dev/null
ERR+=$?
grep net.close $EVT_FILE > /dev/null
ERR+=$?
grep fs.open $EVT_FILE > /dev/null
ERR+=$?
grep fs.close $EVT_FILE > /dev/null
ERR+=$?
grep http.req $EVT_FILE > /dev/null
ERR+=$?
grep http.resp $EVT_FILE > /dev/null
ERR+=$?

# Validate that expected metrics were produced
grep http.req $MTC_FILE > /dev/null
ERR+=$?
grep http.resp $MTC_FILE > /dev/null
ERR+=$?
grep net.tx $MTC_FILE > /dev/null
ERR+=$?
grep net.rx $MTC_FILE > /dev/null
ERR+=$?

# End test
endtest





# Init test
# starttest "nginx http1 tls"

# Run scoped nginx
scope run -- nginx
sleep 2

# Make a http request with curl, to the nginx server
curl --http1.1 -k https://localhost/
sleep 2

# Cleanly exit nginx, allowing metrics and events to be flushed
nginx -s stop
sleep 1

# Validate that expected events were produced
grep net.app $EVT_FILE > /dev/null
ERR+=$?
grep net.open $EVT_FILE > /dev/null
ERR+=$?
grep net.close $EVT_FILE > /dev/null
ERR+=$?
grep fs.open $EVT_FILE > /dev/null
ERR+=$?
grep fs.close $EVT_FILE > /dev/null
ERR+=$?
grep http.req $EVT_FILE > /dev/null
ERR+=$?
grep http.resp $EVT_FILE > /dev/null
ERR+=$?

# Validate that expected metrics were produced
grep http.req $MTC_FILE > /dev/null
ERR+=$?
grep http.resp $MTC_FILE > /dev/null
ERR+=$?
grep net.tx $MTC_FILE > /dev/null
ERR+=$?
grep net.rx $MTC_FILE > /dev/null
ERR+=$?

# End test
endtest





# Init test
# starttest "nginx http2"

# Run scoped nginx
scope run -- nginx
sleep 2

# Make a http request with curl, to the nginx server
curl --http2 -k http://localhost/
sleep 2

# Cleanly exit nginx, allowing metrics and events to be flushed
nginx -s stop
sleep 1

# Validate that expected events were produced
grep net.app $EVT_FILE > /dev/null
ERR+=$?
grep net.open $EVT_FILE > /dev/null
ERR+=$?
grep net.close $EVT_FILE > /dev/null
ERR+=$?
grep fs.open $EVT_FILE > /dev/null
ERR+=$?
grep fs.close $EVT_FILE > /dev/null
ERR+=$?
grep http.req $EVT_FILE > /dev/null
ERR+=$?
grep http.resp $EVT_FILE > /dev/null
ERR+=$?

# Validate that expected metrics were produced
grep http.req $MTC_FILE > /dev/null
ERR+=$?
grep http.resp $MTC_FILE > /dev/null
ERR+=$?
grep net.tx $MTC_FILE > /dev/null
ERR+=$?
grep net.rx $MTC_FILE > /dev/null
ERR+=$?

# End test
endtest





# Init test
# starttest "nginx http2 tls"

# Run scoped nginx
scope run -- nginx
sleep 2

# Make a http request with curl, to the nginx server
curl --http2 -k https://localhost/
sleep 2

# Cleanly exit nginx, allowing metrics and events to be flushed
nginx -s stop
sleep 1

# Validate that expected events were produced
grep net.app $EVT_FILE > /dev/null
ERR+=$?
grep net.open $EVT_FILE > /dev/null
ERR+=$?
grep net.close $EVT_FILE > /dev/null
ERR+=$?
grep fs.open $EVT_FILE > /dev/null
ERR+=$?
grep fs.close $EVT_FILE > /dev/null
ERR+=$?
grep http.req $EVT_FILE > /dev/null
ERR+=$?
grep http.resp $EVT_FILE > /dev/null
ERR+=$?

# Validate that expected metrics were produced
grep http.req $MTC_FILE > /dev/null
ERR+=$?
grep http.resp $MTC_FILE > /dev/null
ERR+=$?
grep net.tx $MTC_FILE > /dev/null
ERR+=$?
grep net.rx $MTC_FILE > /dev/null
ERR+=$?

# End test
endtest





################# RESULTS ################# 

#
# Print test results
#
if (( $FAILED_TEST_COUNT == 0 )); then
    echo ""
    echo ""
    echo "************ ALL NGINX TESTS PASSED ************"
else
    echo "************ SOME NGINX TESTS FAILED ************"
    echo "Failed tests: $FAILED_TEST_LIST"
fi
echo ""

exit ${FAILED_TEST_COUNT}

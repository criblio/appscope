#! /bin/bash

DEBUG=0  # set this to 1 to capture the EVT_FILE for each test

FAILED_TEST_LIST=""
FAILED_TEST_COUNT=0

EVT_FILE="/go/events.log"
touch $EVT_FILE

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
}


#
# plainServer
#
starttest plainServer
cd /go/net
uname -a
ls -al /usr/bin/scope
strings /usr/bin/scope | grep "(Scope Version"
mount | grep shm
file ./plainServer
scope ./plainServer &

# this sleep gives the server a chance to bind to the port (80)
# before we try to hit it with curl
sleep 0.5
curl http://localhost/hello
ERR+=$?

# this sleep gives plainServer a chance to report its events,
# assuming SCOPE_SUMMARY_PERIOD is its default 10s
sleep 10
# This stops plainServer
kill $!

evaltest

grep plainServer $EVT_FILE | grep http-req > /dev/null
ERR+=$?
grep plainServer $EVT_FILE | grep http-resp > /dev/null
ERR+=$?
grep plainServer $EVT_FILE | grep http-metrics > /dev/null
ERR+=$?

endtest


#
# tlsServer
#
starttest tlsServer
cd /go/net
STRUCT_PATH=/go/net/go_offsets.txt
SCOPE_GO_STRUCT_PATH=$STRUCT_PATH scope ./tlsServer &

# this sleep gives the server a chance to bind to the port (4430)
# before we try to hit it with curl
sleep 0.5
curl -k --key server.key --cert server.crt https://localhost:4430/hello
ERR+=$?

# this sleep gives tlsServer a chance to report its events,
# assuming SCOPE_SUMMARY_PERIOD is its default 10s
sleep 10
# This stops tlsServer
kill $!

evaltest

grep tlsServer $EVT_FILE | grep http-req > /dev/null
ERR+=$?
grep tlsServer $EVT_FILE | grep http-resp > /dev/null
ERR+=$?
grep tlsServer $EVT_FILE | grep http-metrics > /dev/null
ERR+=$?

endtest


#
# test_go_struct.sh
#
starttest test_go_struct.sh
cd /go

./test_go_struct.sh $STRUCT_PATH
ERR+=$?

evaltest

touch $EVT_FILE
endtest


#
# fileThread
#
starttest fileThread
cd /go/thread
scope ./fileThread
ERR+=$?
evaltest

grep fileThread $EVT_FILE > /dev/null
ERR+=$?

endtest


#
# Ruby
#
#echo "Creating key files for ruby client and server"
#(cd /opt/test-runner/ruby && openssl req -x509 -newkey rsa:4096 -keyout priv.pem -out cert.pem -days 365 -nodes -subj "/C=US/ST=California/L=San Francisco/O=Cribl/OU=Cribl/CN=localhost")
#starttest Ruby
#RUBY_HTTP_START=$(grep http- $EVT_FILE | grep -c 10101)
#(cd /opt/test-runner/ruby && ./server.rb 10101 &)
#sleep 1
#(cd /opt/test-runner/ruby && ./client.rb 127.0.0.1 10101)
#sleep 1
#evaltest
#RUBY_HTTP_END=$(grep http- $EVT_FILE | grep -c 10101)
#
#if (( $RUBY_HTTP_END - $RUBY_HTTP_START < 6 )); then
#    ERR+=1
#fi
#endtest




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

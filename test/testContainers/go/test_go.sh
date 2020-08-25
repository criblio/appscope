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
scope ./plainServer &

# this sleep gives the server a chance to bind to the port (80)
# before we try to hit it with curl
sleep 0.5
curl http://localhost/hello
ERR+=$?

# This stops plainServer
kill $!

# this sleep gives plainServer a chance to report its events on exit
sleep 0.5


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

# This stops tlsServer
kill $!

# this sleep gives tlsServer a chance to report its events on exit
sleep 0.5

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

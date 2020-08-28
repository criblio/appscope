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
# plainServerDynamic
#
starttest plainServerDynamic
cd /go/net
PORT=80
scope ./plainServerDynamic ${PORT} &

# this sleep gives the server a chance to bind to the port
# before we try to hit it with curl
sleep 0.5
curl http://localhost:${PORT}/hello
ERR+=$?

# This stops plainServerDynamic
pkill -f plainServerDynamic

# this sleep gives plainServerDynamic a chance to report its events on exit
sleep 0.5

evaltest

grep plainServerDynamic $EVT_FILE | grep http-req > /dev/null
ERR+=$?
grep plainServerDynamic $EVT_FILE | grep http-resp > /dev/null
ERR+=$?
grep plainServerDynamic $EVT_FILE | grep http-metrics > /dev/null
ERR+=$?

endtest


#
# plainServerStatic
#
starttest plainServerStatic
cd /go/net
PORT=81
scope ./plainServerStatic ${PORT} &

# this sleep gives the server a chance to bind to the port
# before we try to hit it with curl
sleep 0.5
curl http://localhost:${PORT}/hello
ERR+=$?

# This stops plainServerStatic
kill $!

# this sleep gives plainServerStatic a chance to report its events on exit
sleep 0.5

evaltest

grep plainServerStatic $EVT_FILE | grep http-req > /dev/null
ERR+=$?
grep plainServerStatic $EVT_FILE | grep http-resp > /dev/null
ERR+=$?
grep plainServerStatic $EVT_FILE | grep http-metrics > /dev/null
ERR+=$?

endtest


#
# tlsServerDynamic
#
starttest tlsServerDynamic
cd /go/net
PORT=4430
scope ./tlsServerDynamic ${PORT} &

# this sleep gives the server a chance to bind to the port
# before we try to hit it with curl
sleep 0.5
curl -k --key server.key --cert server.crt https://localhost:${PORT}/hello
ERR+=$?

# This stops tlsServerDynamic
pkill -f tlsServerDynamic

# this sleep gives tlsServerDynamic a chance to report its events on exit
sleep 0.5

evaltest

grep tlsServerDynamic $EVT_FILE | grep http-req > /dev/null
ERR+=$?
grep tlsServerDynamic $EVT_FILE | grep http-resp > /dev/null
ERR+=$?
grep tlsServerDynamic $EVT_FILE | grep http-metrics > /dev/null
ERR+=$?

endtest


#
# tlsServerStatic
#
starttest tlsServerStatic
cd /go/net
PORT=4431
STRUCT_PATH=/go/net/go_offsets.txt
SCOPE_GO_STRUCT_PATH=$STRUCT_PATH scope ./tlsServerStatic ${PORT} &

# this sleep gives the server a chance to bind to the port
# before we try to hit it with curl
sleep 0.5
curl -k --key server.key --cert server.crt https://localhost:${PORT}/hello
ERR+=$?

# This stops tlsServerStatic
kill $!

# this sleep gives tlsServerStatic a chance to report its events on exit
sleep 0.5

evaltest

grep tlsServerStatic $EVT_FILE | grep http-req > /dev/null
ERR+=$?
grep tlsServerStatic $EVT_FILE | grep http-resp > /dev/null
ERR+=$?
grep tlsServerStatic $EVT_FILE | grep http-metrics > /dev/null
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

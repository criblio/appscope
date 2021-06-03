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
# plainServerDynamic
#
starttest plainServerDynamic
cd /go/net
PORT=8080

ldscope ./plainServerDynamic ${PORT} &

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
grep plainServerDynamic $EVT_FILE | grep http-resp | grep "127.0.0.1" > /dev/null
ERR+=$?

evalPayload
ERR+=$?

endtest


#
# plainServerStatic
#
starttest plainServerStatic
cd /go/net
PORT=8081
ldscope ./plainServerStatic ${PORT} &

# this sleep gives the server a chance to bind to the port
# before we try to hit it with curl
sleep 0.5
curl http://localhost:${PORT}/hello
ERR+=$?

# This stops plainServerStatic
pkill -f plainServerStatic

# this sleep gives plainServerStatic a chance to report its events on exit
sleep 0.5

evaltest

grep plainServerStatic $EVT_FILE | grep http-req > /dev/null
ERR+=$?
grep plainServerStatic $EVT_FILE | grep http-resp > /dev/null
ERR+=$?
grep plainServerStatic $EVT_FILE | grep http-metrics > /dev/null
ERR+=$?
grep plainServerStatic $EVT_FILE | grep http-resp | grep "127.0.0.1" > /dev/null
ERR+=$?

evalPayload
ERR+=$?

endtest


#
# tlsServerDynamic
#
starttest tlsServerDynamic
cd /go/net
PORT=4430
ldscope ./tlsServerDynamic ${PORT} &

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
grep tlsServerDynamic $EVT_FILE | grep http-resp | grep "127.0.0.1" > /dev/null
ERR+=$?

evalPayload
ERR+=$?

endtest


#
# tlsServerStatic
#
starttest tlsServerStatic
cd /go/net
PORT=4431
STRUCT_PATH=/go/net/go_offsets.txt
SCOPE_GO_STRUCT_PATH=$STRUCT_PATH ldscope ./tlsServerStatic ${PORT} &

# this sleep gives the server a chance to bind to the port
# before we try to hit it with curl
sleep 0.5
curl -k --key server.key --cert server.crt https://localhost:${PORT}/hello
ERR+=$?

# This stops tlsServerStatic
pkill -f tlsServerStatic

# this sleep gives tlsServerStatic a chance to report its events on exit
sleep 0.5

evaltest

grep tlsServerStatic $EVT_FILE | grep http-req > /dev/null
ERR+=$?
grep tlsServerStatic $EVT_FILE | grep http-resp > /dev/null
ERR+=$?
grep tlsServerStatic $EVT_FILE | grep http-metrics > /dev/null
ERR+=$?
grep tlsServerStatic $EVT_FILE | grep http-resp | grep "127.0.0.1" > /dev/null
ERR+=$?

evalPayload
ERR+=$?

endtest


#
# plainClientDynamic
#
starttest plainClientDynamic
cd /go/net
ldscope ./plainClientDynamic
ERR+=$?

# this sleep gives plainClientDynamic a chance to report its events on exit
sleep 0.5

evaltest

grep plainClientDynamic $EVT_FILE | grep http-req > /dev/null
ERR+=$?
grep plainClientDynamic $EVT_FILE | grep http-resp > /dev/null
ERR+=$?
grep plainClientDynamic $EVT_FILE | grep http-metrics > /dev/null
ERR+=$?

evalPayload
ERR+=$?

endtest


#
# plainClientStatic
#
starttest plainClientStatic
cd /go/net
ldscope ./plainClientStatic
ERR+=$?

# this sleep gives plainClientStatic a chance to report its events on exit
sleep 0.5

evaltest

grep plainClientStatic $EVT_FILE | grep http-req > /dev/null
ERR+=$?
grep plainClientStatic $EVT_FILE | grep http-resp > /dev/null
ERR+=$?
grep plainClientStatic $EVT_FILE | grep http-metrics > /dev/null
ERR+=$?

evalPayload
ERR+=$?

endtest


#
# tlsClientDynamic
#
starttest tlsClientDynamic
cd /go/net
ldscope ./tlsClientDynamic
ERR+=$?

# this sleep gives tlsClientDynamic a chance to report its events on exit
sleep 0.5

evaltest

grep tlsClientDynamic $EVT_FILE | grep http-req > /dev/null
ERR+=$?
grep tlsClientDynamic $EVT_FILE | grep http-resp > /dev/null
ERR+=$?
grep tlsClientDynamic $EVT_FILE | grep http-metrics > /dev/null
ERR+=$?

evalPayload
ERR+=$?

endtest


#
# tlsClientStatic
#
starttest tlsClientStatic
cd /go/net
SCOPE_GO_STRUCT_PATH=$STRUCT_PATH ldscope ./tlsClientStatic
ERR+=$?

# this sleep gives tlsClientStatic a chance to report its events on exit
sleep 0.5

evaltest

grep tlsClientStatic $EVT_FILE | grep http-req > /dev/null
ERR+=$?
grep tlsClientStatic $EVT_FILE | grep http-resp > /dev/null
ERR+=$?
grep tlsClientStatic $EVT_FILE | grep http-metrics > /dev/null
ERR+=$?

evalPayload
ERR+=$?

endtest


#
# test_go_struct_server
#
starttest test_go_struct_server
cd /go

./test_go_struct.sh $STRUCT_PATH /go/net/tlsServerStatic Server
ERR+=$?

evaltest

touch $EVT_FILE
endtest


#
# test_go_struct_client
#
starttest test_go_struct_client
cd /go

./test_go_struct.sh $STRUCT_PATH /go/net/tlsClientStatic Client
ERR+=$?

evaltest

touch $EVT_FILE
endtest


#
# fileThread
#
starttest fileThread
cd /go/thread
ldscope ./fileThread
ERR+=$?
evaltest

grep fileThread $EVT_FILE > /dev/null
ERR+=$?

evalPayload
ERR+=$?

endtest


#
# cgoDynamic
#
starttest cgoDynamic
cd /go/cgo
LD_LIBRARY_PATH=. ldscope ./cgoDynamic
ERR+=$?
evaltest

grep cgoDynamic $EVT_FILE > /dev/null
ERR+=$?

evalPayload
ERR+=$?

endtest


#
# cgoStatic
#
starttest cgoStatic
cd /go/cgo
ldscope ./cgoStatic
ERR+=$?
evaltest

grep cgoStatic $EVT_FILE > /dev/null
ERR+=$?

evalPayload
ERR+=$?

endtest

unset SCOPE_PAYLOAD_ENABLE
unset SCOPE_PAYLOAD_HEADER

#
# Done: print results
#
if (( $FAILED_TEST_COUNT == 0 )); then
    echo ""
    echo ""
    echo "*************** ALL GO TESTS PASSED ***************"
else
    echo "*************** SOME GO TESTS FAILED ***************"
    echo "Failed tests: $FAILED_TEST_LIST"
    echo "Refer to these files for more info:"
    for FAILED_TEST in $FAILED_TEST_LIST; do
        echo "  $EVT_FILE.$FAILED_TEST"
    done
fi
echo ""

exit ${FAILED_TEST_COUNT}

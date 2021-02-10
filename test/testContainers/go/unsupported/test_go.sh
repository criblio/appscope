#! /bin/bash

DEBUG=0  # set this to 1 to capture the LOG_FILE for each test

FAILED_TEST_LIST=""
FAILED_TEST_COUNT=0

LOG_FILE="/go/logs.txt"
touch $LOG_FILE

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

    # copy the LOG_FILE to help with debugging
    if (( $DEBUG )) || [ $RESULT == "FAILED" ]; then
        cp $LOG_FILE $LOG_FILE.$CURRENT_TEST
    fi

    rm $LOG_FILE
}


#
# plainServerDynamic
#
starttest plainServerDynamic
cd /go/net
PORT=80
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

grep "Continuing without AppScope." ${LOG_FILE}
ERR+=$?

endtest


#
# plainServerStatic
#
starttest plainServerStatic
cd /go/net
PORT=81
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

grep "Continuing without AppScope." ${LOG_FILE}
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

grep "Continuing without AppScope." ${LOG_FILE}
ERR+=$?

endtest


#
# tlsServerStatic
#
starttest tlsServerStatic
cd /go/net
PORT=4431
ldscope ./tlsServerStatic ${PORT} &

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

grep "Continuing without AppScope." ${LOG_FILE}
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

grep "Continuing without AppScope." ${LOG_FILE}
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

grep "Continuing without AppScope." ${LOG_FILE}
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

grep "Continuing without AppScope." ${LOG_FILE}
ERR+=$?

endtest


#
# tlsClientStatic
#
starttest tlsClientStatic
cd /go/net
ldscope ./tlsClientStatic
ERR+=$?

# this sleep gives tlsClientStatic a chance to report its events on exit
sleep 0.5

evaltest

grep "Continuing without AppScope." ${LOG_FILE}
ERR+=$?

endtest


#
# fileThread
#
starttest fileThread
cd /go/thread
ldscope ./fileThread
ERR+=$?
evaltest

grep "Continuing without AppScope." ${LOG_FILE}
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

grep "Continuing without AppScope." ${LOG_FILE}
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

grep "Continuing without AppScope." ${LOG_FILE}
ERR+=$?

endtest




#
# Done: print results
#
if (( $FAILED_TEST_COUNT == 0 )); then
    echo ""
    echo ""
    echo "*************** ALL TESTS PASSED ***************"
else
    echo "*************** SOME TESTS FAILED ***************"
    echo "Failed tests: $FAILED_TEST_LIST"
    echo "Refer to these files for more info:"
    for FAILED_TEST in $FAILED_TEST_LIST; do
        echo "  $LOG_FILE.$FAILED_TEST"
    done
fi

exit ${FAILED_TEST_COUNT}

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
scope ./plainServerDynamic ${PORT}
SCOPE_RET=$?

evaltest

# we expect a non-zero return code.
if [ $SCOPE_RET -eq 0 ]; then
    ERR+=1
fi

endtest


#
# plainServerStatic
#
starttest plainServerStatic
cd /go/net
PORT=81
scope ./plainServerStatic ${PORT}
SCOPE_RET=$?

evaltest

# we expect a non-zero return code.
if [ $SCOPE_RET -eq 0 ]; then
    ERR+=1
fi

endtest


#
# tlsServerDynamic
#
starttest tlsServerDynamic
cd /go/net
PORT=4430
scope ./tlsServerDynamic ${PORT}
SCOPE_RET=$?

evaltest

# we expect a non-zero return code.
if [ $SCOPE_RET -eq 0 ]; then
    ERR+=1
fi

endtest


#
# tlsServerStatic
#
starttest tlsServerStatic
cd /go/net
PORT=4431
scope ./tlsServerStatic ${PORT}
SCOPE_RET=$?

evaltest

# we expect a non-zero return code.
if [ $SCOPE_RET -eq 0 ]; then
    ERR+=1
fi

endtest


#
# plainClientDynamic
#
starttest plainClientDynamic
cd /go/net
scope ./plainClientDynamic
SCOPE_RET=$?

evaltest

# we expect a non-zero return code.
if [ $SCOPE_RET -eq 0 ]; then
    ERR+=1
fi

endtest


#
# plainClientStatic
#
starttest plainClientStatic
cd /go/net
scope ./plainClientStatic
SCOPE_RET=$?

evaltest

# we expect a non-zero return code.
if [ $SCOPE_RET -eq 0 ]; then
    ERR+=1
fi

endtest


#
# tlsClientDynamic
#
starttest tlsClientDynamic
cd /go/net
scope ./tlsClientDynamic
SCOPE_RET=$?

evaltest

# we expect a non-zero return code.
if [ $SCOPE_RET -eq 0 ]; then
    ERR+=1
fi

endtest


#
# tlsClientStatic
#
starttest tlsClientStatic
cd /go/net
scope ./tlsClientStatic
SCOPE_RET=$?

evaltest

# we expect a non-zero return code.
if [ $SCOPE_RET -eq 0 ]; then
    ERR+=1
fi

endtest


#
# fileThread
#
starttest fileThread
cd /go/thread
scope ./fileThread
SCOPE_RET=$?

evaltest

# we expect a non-zero return code.
if [ $SCOPE_RET -eq 0 ]; then
    ERR+=1
fi

endtest


#
# cgoDynamic
#
starttest cgoDynamic
cd /go/cgo
LD_LIBRARY_PATH=. scope ./cgoDynamic
SCOPE_RET=$?

evaltest

# we expect a non-zero return code.
if [ $SCOPE_RET -eq 0 ]; then
    ERR+=1
fi

endtest


#
# cgoStatic
#
starttest cgoStatic
cd /go/cgo
scope ./cgoStatic
SCOPE_RET=$?

evaltest

# we expect a non-zero return code.
if [ $SCOPE_RET -eq 0 ]; then
    ERR+=1
fi

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
        echo "  $EVT_FILE.$FAILED_TEST"
    done
fi

exit ${FAILED_TEST_COUNT}

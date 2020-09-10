#!/bin/bash

DEBUG=0  # set this to 1 to capture the EVT_FILE for each test

FAILED_TEST_LIST=""
FAILED_TEST_COUNT=0

EVT_FILE="/gogen/events.log"
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
# gogen
#
starttest gogen
cd /gogen

# start something to receive the network activity
#
# tcpserver is an app built from git@bitbucket.org:cribl/scope.git
# gcc -g test/manual/tcpserver.c -lpthread -o tcpserver
RX_FILE="/gogen/receive.log"
tcpserver 12345 > $RX_FILE &

# Run gogen
scope ./gogen -c coccyx/weblog -ot json -o tcp --url localhost:12345 -g 10 --os 10 -v gen -c 10 -ei 10000 -i 60
ERR+=$?
evaltest

grep gogen $EVT_FILE > /dev/null
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


#! /bin/bash

DEBUG=0  # set this to 1 to capture the EVT_FILE for each test

FAILED_TEST_LIST=""
FAILED_TEST_COUNT=0

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

}


################# START TESTS ################# 

#
# attach
#
starttest attach

set -x

# run sleep
sleep 1000 & 
sleep_pid=$!

# attach to sleep process
scope attach $sleep_pid
ERR+=$?

# wait for attach to execute, then end sleep process
sleep 2
kill $sleep_pid

# assert .scope directory exists
ls .scope
ERR+=$?

# assert .scope/history directory exists
ls .scope/history
ERR+=$?

# assert sleep session files exist
# TODO

# assert sleep config files exist
# TODO

set +x

evaltest

endtest


################# END TESTS ################# 

unset SCOPE_PAYLOAD_ENABLE
unset SCOPE_PAYLOAD_HEADER

#
# Done: print results
#
if (( $FAILED_TEST_COUNT == 0 )); then
    echo ""
    echo ""
    echo "*************** ALL ATTACH TESTS PASSED ***************"
else
    echo "*************** SOME ATTACH TESTS FAILED ***************"
    echo "Failed tests: $FAILED_TEST_LIST"
fi
echo ""

exit ${FAILED_TEST_COUNT}

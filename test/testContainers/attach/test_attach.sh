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

# Run sleep
sleep 1000 & 
sleep_pid=$!

# Attach to sleep process
scope attach $sleep_pid
ERR+=$?

# Wait for attach to execute, then end sleep process
sleep 2
kill $sleep_pid

# Navigate to home directory
cd

# Assert .scope directory exists
if [ -d .scope ]; then
	echo "PASS Scope directory exists"
else 
	echo "FAIL Scope directory missing"
	ERR+=1
fi

# Assert sleep session directory exists (in /tmp)
if [ -d /tmp/$sleep_pid* ]; then
	echo "PASS Scope sleep session directory exists"
else 
	echo "FAIL Scope sleep session directory missing"
	ERR+=1
fi

# Assert sleep config file exists
if [ -f /tmp/"$sleep_pid"*/scope.yml ]; then
	echo "PASS Scope sleep session scope.yml exists"
else 
	echo "FAIL Scope sleep session scope.yml missing"
	ERR+=1
fi

# Compare sleep config.yml with expected.yml
cat /tmp/"$sleep_pid"*/scope.yml | sed -e 's/"$sleep_pid"_1_[0-9]_[0-9]*/"$sleep_pid"_1_SESSIONPATH/' | diff - /expected.yml
if [ $? -eq 0 ]; then
	echo "PASS Scope sleep config as expected"
else
	echo "FAIL Scope sleep config not as expected"
	ERR+=1
fi

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

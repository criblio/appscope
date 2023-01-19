#! /bin/bash
DEBUG=0  # set this to 1 to capture the EVT_FILE for each test
FAILED_TEST_LIST=""
FAILED_TEST_COUNT=0
INSPECT_FILE="/opt/test/inspect.txt"
CRIBL_SOCKET="/opt/cribl/state/appscope.sock"

starttest(){
    CURRENT_TEST=$1
    echo "=============================================="
    echo "             Testing $CURRENT_TEST            "
    echo "=============================================="
    ERR=0
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

    echo "******************* $RESULT *******************"
    echo ""
    echo ""
}

run() {
    CMD="$@"
    echo "\`${CMD}\`"
    OUT=$(${CMD} 2>&1)
    RET=$?
}

################# START TESTS ################# 

starttest test_inspect_cribl_enable

# First test that we are not connected
PRE_SCOPE_CRIBL_ENABLE=$SCOPE_CRIBL_ENABLE
unset SCOPE_CRIBL_ENABLE

LD_PRELOAD=/usr/local/scope/lib/libscope.so python3 -m http.server 1> /dev/null 2> /dev/null &
sleep 2
export SCOPE_CRIBL_ENABLE=$PRE_SCOPE_CRIBL_ENABLE
PYTHON_PID=`pidof python3`

scope inspect $PYTHON_PID > $INSPECT_FILE
if [ $? != 0 ]; then
    echo "first inspect fails"
    cat $INSPECT_FILE
    ERR+=1
fi
sleep 2

jq '.interfaces | .[] | select(.name=="log") | .connected' $INSPECT_FILE | grep -q "true"
if [ $? != 0 ]; then
    echo "inspect cribl connected fails expected enable logs"
    cat $INSPECT_FILE
    ERR+=1
fi

jq '.interfaces | .[] | select(.name=="cribl") | .connected' $INSPECT_FILE | grep -q "false"
if [ $? != 0 ]; then
    echo "inspect cribl connected fails expected disable cribl"
    cat $INSPECT_FILE
    ERR+=1
fi

count=$(jq '.interfaces | length' $INSPECT_FILE)
if [ $count -ne 2 ]; then
    echo "inspect cribl connected fails expected only two interfaces"
    cat $INSPECT_FILE
    ERR+=1
fi

rm $INSPECT_FILE

# Test that we are not connected
nc -lU $CRIBL_SOCKET 1> /dev/null 2> /dev/null &
NC_PID=`pidof nc`

# Give time to connect
sleep 5

scope inspect $PYTHON_PID > $INSPECT_FILE
if [ $? != 0 ]; then
    echo "second inspect fails"
    cat $INSPECT_FILE
    ERR+=1
fi

jq '.interfaces | .[] | select(.name=="log") | .connected' $INSPECT_FILE | grep -q "true"
if [ $? != 0 ]; then
    echo "inspect cribl connected fails expected enable logs"
    cat $INSPECT_FILE
    ERR+=1
fi

jq '.interfaces | .[] | select(.name=="cribl") | .connected' $INSPECT_FILE | grep -q "true"
if [ $? != 0 ]; then
    echo "inspect cribl connected fails expected enable cribl"
    cat $INSPECT_FILE
    ERR+=1
fi

count=$(jq '.interfaces | length' $INSPECT_FILE)
if [ $count -ne 2 ]; then
    echo "inspect cribl connected fails expected only two interfaces"
    cat $INSPECT_FILE
    ERR+=1
fi

rm $INSPECT_FILE

kill -9 $PYTHON_PID 1> /dev/null 2> /dev/null
kill -9 $NC_PID 1> /dev/null 2> /dev/null

endtest

starttest test_inspect_cribl_disable

# First test that we are not connected
SCOPE_CRIBL_ENABLE=false LD_PRELOAD=/usr/local/scope/lib/libscope.so python3 -m http.server 1> /dev/null 2> /dev/null &
sleep 2
PYTHON_PID=`pidof python3`

scope inspect $PYTHON_PID > $INSPECT_FILE
if [ $? != 0 ]; then
    echo "first inspect fails"
    cat $INSPECT_FILE
    ERR+=1
fi
sleep 2

jq '.interfaces | .[] | select(.name=="log") | .connected' $INSPECT_FILE | grep -q "true"
if [ $? != 0 ]; then
    echo "inspect cribl connected fails expected enable logs"
    cat $INSPECT_FILE
    ERR+=1
fi

jq '.interfaces | .[] | select(.name=="events") | .connected' $INSPECT_FILE | grep -q "false"
if [ $? != 0 ]; then
    echo "inspect cribl connected fails expected disable events"
    cat $INSPECT_FILE
    ERR+=1
fi

jq '.interfaces | .[] | select(.name=="metrics") | .connected' $INSPECT_FILE | grep -q "true"
if [ $? != 0 ]; then
    echo "inspect cribl connected fails expected enable metrics"
    cat $INSPECT_FILE
    ERR+=1
fi

count=$(jq '.interfaces | length' $INSPECT_FILE)
if [ $count -ne 3 ]; then
    echo "inspect cribl connected fails expected only three interfaces"
    cat $INSPECT_FILE
    ERR+=1
fi

rm $INSPECT_FILE

# Test that we are not connected
nc -l -p 9109 1> /dev/null 2> /dev/null &
NC_PID=`pidof nc`

# Give time to connect
sleep 5

scope inspect $PYTHON_PID > $INSPECT_FILE
if [ $? != 0 ]; then
    echo "second inspect fails"
    cat $INSPECT_FILE
    ERR+=1
fi

jq '.interfaces | .[] | select(.name=="log") | .connected' $INSPECT_FILE | grep -q "true"
if [ $? != 0 ]; then
    echo "inspect cribl connected fails expected enable logs"
    cat $INSPECT_FILE
    ERR+=1
fi

jq '.interfaces | .[] | select(.name=="events") | .connected' $INSPECT_FILE | grep -q "true"
if [ $? != 0 ]; then
    echo "inspect cribl connected fails expected enable events"
    cat $INSPECT_FILE
    ERR+=1
fi

jq '.interfaces | .[] | select(.name=="metrics") | .connected' $INSPECT_FILE | grep -q "true"
if [ $? != 0 ]; then
    echo "inspect cribl connected fails expected enable metrics"
    cat $INSPECT_FILE
    ERR+=1
fi

count=$(jq '.interfaces | length' $INSPECT_FILE)
if [ $count -ne 3 ]; then
    echo "inspect cribl connected fails expected only two interfaces"
    cat $INSPECT_FILE
    ERR+=1
fi

rm $INSPECT_FILE

kill -9 $PYTHON_PID 1> /dev/null 2> /dev/null
kill -9 $NC_PID 1> /dev/null 2> /dev/null

endtest

# Below tests are based on integration/payload
starttest test_payload_off

SCOPE_CRIBL_ENABLE=false SCOPE_PAYLOAD_TO_DISK=true LD_PRELOAD=/usr/local/scope/lib/libscope.so python3 -m http.server 1> /dev/null 2> /dev/null &
sleep 2
PYTHON_PID=`pidof python3`

scope inspect $PYTHON_PID > $INSPECT_FILE

jq '.interfaces | .[] | .name ' $INSPECT_FILE | grep payload
if [ $? == 0 ]; then
    echo "inspect cribl connected fails expected no payload section"
    cat $INSPECT_FILE
    ERR+=1
fi

rm $INSPECT_FILE

kill -9 $PYTHON_PID 1> /dev/null 2> /dev/null

endtest

starttest test_payload_on_to_cribl

PRE_SCOPE_CRIBL_ENABLE=$SCOPE_CRIBL_ENABLE
unset SCOPE_CRIBL_ENABLE
SCOPE_PAYLOAD_ENABLE=true LD_PRELOAD=/usr/local/scope/lib/libscope.so python3 -m http.server 1> /dev/null 2> /dev/null &
export SCOPE_CRIBL_ENABLE=$PRE_SCOPE_CRIBL_ENABLE

sleep 2
PYTHON_PID=`pidof python3`

scope inspect $PYTHON_PID > $INSPECT_FILE

jq '.interfaces | .[] | select(.name=="payload") | .connected' $INSPECT_FILE | grep -q "false"
if [ $? != 0 ]; then
    echo "inspect cribl connected fails expected disable payload"
    cat $INSPECT_FILE
    ERR+=1
fi

jq '.interfaces | .[] | select(.name=="payload") | .config' $INSPECT_FILE | grep -q "edge"
if [ $? != 0 ]; then
    echo "inspect cribl connected fails expected payload should point to edge"
    cat $INSPECT_FILE
    ERR+=1
fi

rm $INSPECT_FILE

kill -9 $PYTHON_PID 1> /dev/null 2> /dev/null

endtest

# Uncomment this when detection of payload to disk is supported - see `payloadTransportEnabled`
# starttest test_payload_on_to_cribl_overriden_to_disk

# PRE_SCOPE_CRIBL_ENABLE=$SCOPE_CRIBL_ENABLE
# unset SCOPE_CRIBL_ENABLE
# SCOPE_PAYLOAD_ENABLE=true SCOPE_PAYLOAD_TO_DISK=true LD_PRELOAD=/usr/local/scope/lib/libscope.so python3 -m http.server 1> /dev/null 2> /dev/null &
# export SCOPE_CRIBL_ENABLE=$PRE_SCOPE_CRIBL_ENABLE

# sleep 2
# PYTHON_PID=`pidof python3`

# scope inspect $PYTHON_PID > $INSPECT_FILE

# jq '.interfaces | .[] | select(.name=="payload") | .connected' $INSPECT_FILE | grep -q "true"
# if [ $? != 0 ]; then
#     echo "inspect cribl connected fails expected enable payload"
#     cat $INSPECT_FILE
#     ERR+=1
# fi

# jq '.interfaces | .[] | select(.name=="payload") | .config' $INSPECT_FILE | grep -q "edge"
# if [ $? == 0 ]; then
#     echo "inspect cribl connected fails expected payload should not point to edge"
#     cat $INSPECT_FILE
#     ERR+=1
# fi

# rm $INSPECT_FILE

# kill -9 $PYTHON_PID 1> /dev/null 2> /dev/null

# endtest

################# END TESTS ################# 

#
# Print test results
#
if (( $FAILED_TEST_COUNT == 0 )); then
    echo ""
    echo ""
    echo "************ ALL CLI TESTS PASSED ************"
else
    echo "************ SOME CLI TESTS FAILED ************"
    echo "Failed tests: $FAILED_TEST_LIST"
fi
echo ""

exit ${FAILED_TEST_COUNT}

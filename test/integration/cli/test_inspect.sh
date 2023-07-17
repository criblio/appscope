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

# Check expected interface property value
interface_prop_check() {
    local interface_name=$1
    local interface_prop=$2
    local expected_state=$3
    jq '.interfaces[] | select(.name=='\"$interface_name\"') | .'$interface_prop'' $INSPECT_FILE  | grep -q "$expected_state"
    if [ $? != 0 ]; then
        echo "interface_check fails, params: $interface_name $interface_prop $expected_state"
        cat $INSPECT_FILE
        ERR+=1
    fi
}

# Check expected number of interfaces
interface_length_check() {
    local expected_length=$1
    count=$(jq '.interfaces | length' $INSPECT_FILE)
    if [ $count -ne $expected_length ]; then
        echo "interface_length_check, params $expected_length"
        cat $INSPECT_FILE
        ERR+=1
    fi
}

# Check expected number of responses, and that they all contain interfaces
responses_length_check() {
    local expected_length=$1
    count=$(jq '[.[] | .interfaces] | length' $INSPECT_FILE)
    if [ $count -ne $expected_length ]; then
        echo "responses_length_check, params $expected_length"
        cat $INSPECT_FILE
        ERR+=1
    fi
}

# Call scope inspect and redirect to file
inspect_file_redirect_to_file() {
    local pid=$1
    scope inspect $pid > $INSPECT_FILE
    if [ $? != 0 ]; then
        echo "inspect_file_redirect_to_file fails, params $pid"
        cat $INSPECT_FILE
        ERR+=1
    fi
    # Time to save file
    sleep 2
}

# Call scope inspect --all and redirect to file
inspect_all_file_redirect_to_file() {
    scope inspect --all > $INSPECT_FILE
    if [ $? != 0 ]; then
        echo "inspect_all_file_redirect_to_file fails"
        cat $INSPECT_FILE
        ERR+=1
    fi
    # Time to save file
    sleep 2
}

################# START TESTS #################

# default values: expected to see log, cribl
starttest test_inspect_default_values

# First test that we are not connected
PRE_SCOPE_CRIBL_ENABLE=$SCOPE_CRIBL_ENABLE
unset SCOPE_CRIBL_ENABLE

LD_PRELOAD=/usr/local/scope/lib/libscope.so python3 -m http.server 1> /dev/null 2> /dev/null &
sleep 2
export SCOPE_CRIBL_ENABLE=$PRE_SCOPE_CRIBL_ENABLE
PYTHON_PID=`pidof python3`

inspect_file_redirect_to_file $PYTHON_PID

interface_prop_check "log" "connected" "true"
interface_prop_check "cribl" "connected" "false"
interface_prop_check "cribl" "config" "edge"
interface_length_check 2

rm $INSPECT_FILE

# Test that we are not connected
nc -lU $CRIBL_SOCKET 1> /dev/null 2> /dev/null &
NC_PID=`pidof nc`

# Give time to connect
sleep 6
inspect_file_redirect_to_file $PYTHON_PID

interface_prop_check "log" "connected" "true"
interface_prop_check "cribl" "connected" "true"
interface_prop_check "cribl" "config" "edge"
interface_length_check 2

rm $INSPECT_FILE

kill -9 $PYTHON_PID 1> /dev/null 2> /dev/null
kill -9 $NC_PID 1> /dev/null 2> /dev/null

endtest


# inspect_all: expected to see log, cribl for each process
starttest test_inspect_all

# First test that we are not connected
PRE_SCOPE_CRIBL_ENABLE=$SCOPE_CRIBL_ENABLE
unset SCOPE_CRIBL_ENABLE

# Start 2 processes
LD_PRELOAD=/usr/local/scope/lib/libscope.so python3 -m http.server 1> /dev/null 2> /dev/null &
sleep 2
export SCOPE_CRIBL_ENABLE=$PRE_SCOPE_CRIBL_ENABLE
PYTHON_PID=`pidof python3`

LD_PRELOAD=/usr/local/scope/lib/libscope.so sleep 1000 1> /dev/null 2> /dev/null &
sleep 2
export SCOPE_CRIBL_ENABLE=$PRE_SCOPE_CRIBL_ENABLE
SLEEP_PID=`pidof sleep`

inspect_all_file_redirect_to_file

responses_length_check 2

rm $INSPECT_FILE

# Test that we are not connected
nc -lU $CRIBL_SOCKET 1> /dev/null 2> /dev/null &
NC_PID=`pidof nc`

kill -9 $PYTHON_PID 1> /dev/null 2> /dev/null
kill -9 $SLEEP_PID 1> /dev/null 2> /dev/null
kill -9 $NC_PID 1> /dev/null 2> /dev/null

endtest


# disabled cribl expected to see log, events, metrics
starttest test_inspect_cribl_disable_payload_disable

SCOPE_CRIBL_ENABLE=false LD_PRELOAD=/usr/local/scope/lib/libscope.so python3 -m http.server 1> /dev/null 2> /dev/null &
sleep 2
PYTHON_PID=`pidof python3`

inspect_file_redirect_to_file $PYTHON_PID

interface_prop_check "log" "connected" "true"
interface_prop_check "events" "connected" "false"
interface_prop_check "metrics" "connected" "true"
interface_length_check 3

rm $INSPECT_FILE

# # Test that we are not connected
nc -l -p 9109 1> /dev/null 2> /dev/null &
NC_PID=`pidof nc`

# # Give time to connect
sleep 6

inspect_file_redirect_to_file $PYTHON_PID

interface_prop_check "log" "connected" "true"
interface_prop_check "events" "connected" "true"
interface_prop_check "metrics" "connected" "true"
interface_length_check 3

rm $INSPECT_FILE

kill -9 $PYTHON_PID 1> /dev/null 2> /dev/null
kill -9 $NC_PID 1> /dev/null 2> /dev/null

endtest

# disabled cribl, enabled payloads enabled (via env var) expected to see log, events, metrics, payload
starttest test_inspect_cribl_disable_payload_enable_via_env

SCOPE_CRIBL_ENABLE=false SCOPE_PAYLOAD_ENABLE=true LD_PRELOAD=/usr/local/scope/lib/libscope.so python3 -m http.server 1> /dev/null 2> /dev/null &
sleep 2
PYTHON_PID=`pidof python3`

inspect_file_redirect_to_file $PYTHON_PID

interface_prop_check "log" "connected" "true"
interface_prop_check "events" "connected" "false"
interface_prop_check "metrics" "connected" "true"
interface_prop_check "payload" "connected" "true"
interface_prop_check "payload" "config" "dir:///tmp"
interface_length_check 4

rm $INSPECT_FILE

kill -9 $PYTHON_PID 1> /dev/null 2> /dev/null

endtest

# in config disabled cribl payloads enabled (via protocol list) expected to see log, events, metrics, payload
starttest test_inspect_cribl_disable_payload_enable_via_config

SCOPE_CONF_PATH=/opt/test/bin/payload_conf.yml LD_PRELOAD=/usr/local/scope/lib/libscope.so python3 -m http.server 1> /dev/null 2> /dev/null &
sleep 2
PYTHON_PID=`pidof python3`

inspect_file_redirect_to_file $PYTHON_PID

interface_prop_check "log" "connected" "true"
interface_prop_check "events" "connected" "false"
interface_prop_check "metrics" "connected" "true"
interface_prop_check "payload" "connected" "true"
interface_prop_check "payload" "config" "dir:///tmp"
interface_length_check 4

rm $INSPECT_FILE

kill -9 $PYTHON_PID 1> /dev/null 2> /dev/null

endtest

# enabled cribl, payload_to_disk enable, expected to see log, events, metrics
starttest test_inspect_cribl_enable_payload_disable

SCOPE_PAYLOAD_TO_DISK=true LD_PRELOAD=/usr/local/scope/lib/libscope.so python3 -m http.server 1> /dev/null 2> /dev/null &
sleep 2
PYTHON_PID=`pidof python3`

inspect_file_redirect_to_file $PYTHON_PID

interface_prop_check "log" "connected" "true"
interface_prop_check "events" "connected" "false"
interface_prop_check "metrics" "connected" "true"
interface_length_check 3

rm $INSPECT_FILE

kill -9 $PYTHON_PID 1> /dev/null 2> /dev/null

endtest

# disabled cribl, payload enable, expected to see log, cribl, payload
starttest test_inspect_cribl_enable_payload_enable

PRE_SCOPE_CRIBL_ENABLE=$SCOPE_CRIBL_ENABLE
unset SCOPE_CRIBL_ENABLE
SCOPE_PAYLOAD_ENABLE=true LD_PRELOAD=/usr/local/scope/lib/libscope.so python3 -m http.server 1> /dev/null 2> /dev/null &
export SCOPE_CRIBL_ENABLE=$PRE_SCOPE_CRIBL_ENABLE
sleep 2
PYTHON_PID=`pidof python3`

inspect_file_redirect_to_file $PYTHON_PID

interface_prop_check "log" "connected" "true"
interface_prop_check "cribl" "connected" "false"
interface_prop_check "cribl" "config" "edge"
interface_prop_check "payload" "connected" "false"
interface_prop_check "payload" "config" "edge"
interface_length_check 3

rm $INSPECT_FILE

kill -9 $PYTHON_PID 1> /dev/null 2> /dev/null

endtest

# enabled cribl, payload enable, payload_to_disk enable, expected to see log, cribl, payload
starttest test_inspect_cribl_enable_payload_enable_payload_to_disk_enable

PRE_SCOPE_CRIBL_ENABLE=$SCOPE_CRIBL_ENABLE
unset SCOPE_CRIBL_ENABLE
SCOPE_PAYLOAD_TO_DISK=true SCOPE_PAYLOAD_ENABLE=true LD_PRELOAD=/usr/local/scope/lib/libscope.so python3 -m http.server 1> /dev/null 2> /dev/null &
export SCOPE_CRIBL_ENABLE=$PRE_SCOPE_CRIBL_ENABLE
sleep 2
PYTHON_PID=`pidof python3`

inspect_file_redirect_to_file $PYTHON_PID

interface_prop_check "log" "connected" "true"
interface_prop_check "cribl" "connected" "false"
interface_prop_check "cribl" "config" "edge"
interface_prop_check "payload" "connected" "true"
interface_prop_check "payload" "config" "dir:///tmp"
interface_length_check 3

rm $INSPECT_FILE

kill -9 $PYTHON_PID 1> /dev/null 2> /dev/null

endtest

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

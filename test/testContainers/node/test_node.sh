#! /bin/bash

DEBUG=0  # set this to 1 to capture the EVT_FILE for each test
FAILED_TEST_LIST=""
FAILED_TEST_COUNT=0

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

outputs() {
    if ! grep "$1" <<< "$OUT" >/dev/null; then
        echo "FAIL: Expected \"$1\" in output of \`$CMD\`, got $OUT"
        ERR+=1
    else
	echo "PASS: Output as expected"
    fi
}

doesnt_output() {
    if grep "$1" <<< "$OUT" >/dev/null; then
        echo "FAIL: Didn't expect \"$1\" in output of \`$CMD\`"
        ERR+=1
    else
	echo "PASS: Output as expected"
    fi
}

is_file() {
    if [ ! -f "$1" ] ; then
        echo "FAIL: File $1 does not exist"
        ERR+=1
    else
	echo "PASS: File exists"
    fi
}

is_dir() {
    if [ ! -d "$1" ] ; then
        echo "FAIL: Directory $1 does not exist"
        ERR+=1
    else
	echo "PASS: Directory exists"
    fi
}

returns() {
    if [ "$RET" != "$1" ]; then
        echo "FAIL: Expected \`$CMD\` to return $1, got $RET"
        ERR+=1
    else
	echo "PASS: Return value as expected"
    fi
}





################# START TESTS ################# 

#
# npm ci
#
starttest "npm ci"

# Run tcpserver
RX_FILE="/receive.log"
tcpserver 9000 > $RX_FILE &

# Scope npm ci
cd /
run scope run -c tcp://127.0.0.1:9000 -- npm ci
returns 0

endtest





################# END TESTS ################# 

#
# Print test results
#
if (( $FAILED_TEST_COUNT == 0 )); then
    echo ""
    echo ""
    echo "************ ALL NODE TESTS PASSED ************"
else
    echo "************ SOME NODE TESTS FAILED ************"
    echo "Failed tests: $FAILED_TEST_LIST"
fi
echo ""

exit ${FAILED_TEST_COUNT}

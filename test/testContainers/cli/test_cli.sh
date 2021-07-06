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
# Attach by pid
#
starttest "Attach by pid"

# Run sleep
sleep 1000 & 
sleep_pid=$!

# Attach to sleep process
run scope attach $sleep_pid
returns 0

# Wait for attach to execute, then end sleep process
sleep 2
kill $sleep_pid

# Assert .scope directory exists
is_dir /root/.scope

# Assert sleep session directory exists (in /tmp)
is_dir /tmp/${sleep_pid}*

# Assert sleep config file exists
is_file /tmp/${sleep_pid}*/scope.yml

# Compare sleep config.yml with expected.yml
cat /tmp/${sleep_pid}*/scope.yml | sed -e "s/${sleep_pid}_1_[0-9][0-9]*_[0-9]*/SESSIONPATH/" | diff - /expected.yml
if [ $? -eq 0 ]; then
	echo "PASS: Scope sleep config as expected"
else
	echo "FAIL: Scope sleep config not as expected"
	ERR+=1
fi

endtest





#
# Attach by name
#
starttest "Attach by name"

# Run sleep
sleep 1000 & 
sleep_pid=$!

# Attach to sleep process
run scope attach sleep
outputs "Attaching to process ${sleep_pid}"
returns 0

endtest





#
# Scope ps
#
starttest "Scope ps"

# Scope ps
run scope ps
outputs "ID	PID	USER	COMMAND
1	${sleep_pid} 	root	sleep"
returns 0


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

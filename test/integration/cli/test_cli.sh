#! /bin/bash
export SCOPE_EVENT_DEST=file:///opt/test/logs/events.log

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

scopedProcessNumber() {
    local procFound=$(($(scope ps | wc -l) - 1 ))

    echo $procFound
}

wordPresentInFile() {
    local word=$1
    local fileName=$2

    count=$(grep $word $fileName | wc -l)
    if [ $count -eq 0 ] ; then
        echo "FAIL: Value $1 is not present int $2"
        ERR+=1
    else
	    echo "PASS: Value $1 present in $2"
    fi
}

# wait maximum 30 seconds
waitForCmdscopedProcessNumber() {
    local expScoped=$1
    local retry=0
    local maxRetry=30
    local delay=1
    until [ "$retry" -ge "$maxRetry" ]
    do
        count=$(scopedProcessNumber)
        if [ "$count" -eq "$expScoped" ] ; then
            return
        fi
        retry=$((retry+1)) 
        sleep "$delay"
    done
    echo "FAIL: waiting for the number $expScoped scoped process $count"
    ERR+=1
}

################# START TESTS ################# 

#
# Attach by pid
#
starttest "Attach by pid"

# Run sleep
sleep 1000 & 
sleep_pid=$!

sleep 1

# Attach to sleep process
run scope attach $sleep_pid
returns 0

# Wait for attach to execute
waitForCmdscopedProcessNumber 1

# Detach to sleep process by PID
run scope detach $sleep_pid
outputs "Detaching from pid ${sleep_pid}"
returns 0

# Wait for detach to execute
waitForCmdscopedProcessNumber 0

# Reattach to sleep process by PID
run scope attach $sleep_pid
outputs "Reattaching to pid ${sleep_pid}"
returns 0

# Wait for reattach to execute
waitForCmdscopedProcessNumber 1

# End sleep process
kill $sleep_pid

# Assert .scope directory exists
is_dir /root/.scope

# Assert sleep session directory exists (in /tmp)
is_dir /tmp/${sleep_pid}*

# Assert sleep config file exists
is_file /tmp/${sleep_pid}*/scope.yml

# Compare sleep config.yml files (attach and reattach) with expected.yml
for scopedirpath in /tmp/${sleep_pid}_*; do
    scopedir=$(basename "$scopedirpath")
    cat $scopedirpath/scope.yml | sed -e "s/$scopedir/SESSIONPATH/" | diff - /expected.yml
    if [ $? -eq 0 ]; then
        echo "PASS: Scope sleep config as expected"
    else
        echo "FAIL: Scope sleep config not as expected"
        ERR+=1
    fi
done

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

# Wait for attach to execute
waitForCmdscopedProcessNumber 1

# Scope ps
run scope ps
outputs "ID	PID	USER	COMMAND
1	${sleep_pid} 	root	sleep 1000"
returns 0

endtest


#
# Scope start no force
#
starttest "Scope start no force"

# Scope start
run scope start
outputs "If you wish to proceed, run again with the -f flag."
returns 0

endtest


#
# Scope start no input
#
starttest "Scope start no input"

# Scope start
run scope start -f
outputs "Exiting due to start failure"
returns 1
scope logs -s | grep -q "Missing filter data"
ERR+=$?

endtest


#
# Scope start empty file pipeline
#
starttest "Scope start empty file pipeline"

OUT=$(cat /opt/test-runner/empty_file | scope start -f 2>&1)
RET=$?
outputs "Exiting due to start failure"
returns 1
scope logs -s | grep -q "Missing filter data"
ERR+=$?

endtest


#
# Scope start empty file redirect
#
starttest "Scope start empty file redirect"

OUT=$(scope start -f < /opt/test-runner/empty_file 2>&1)
RET=$?
outputs "Exiting due to start failure"
scope logs -s | grep -q "Missing filter data"
ERR+=$?
returns 1

endtest


#
# Scope detach by name
#
starttest "Scope detach by name"

run scope detach sleep
outputs "Detaching from pid ${sleep_pid}"
returns 0

endtest

# Give time to consume configuration file (without sleep)
timeout 4s tail -f /dev/null


#
# Scope reattach by name
#
starttest "Scope reattach by name"

# reattach by name
run scope attach sleep
outputs "Reattaching to pid ${sleep_pid}"
returns 0

# Kill sleep process
kill $sleep_pid

endtest


#
# Scope detach all
#
starttest "Scope detach all"

# Run sleep
sleep 1000 & 
sleep_pid1=$!

# Run another sleep
sleep 1000 & 
sleep_pid2=$!

# Attach to sleep processes
run scope attach $sleep_pid1
returns 0
run scope attach $sleep_pid2
returns 0

# Wait for attach to execute
waitForCmdscopedProcessNumber 2

# Detach from sleep processes
yes | scope detach --all 2>&1
RET=$?
returns 0

endtest


##
## Scope daemon
##
#starttest "Scope daemon"
#
## Start a netcat listener
#nc -l -p 9109 > crash.out &
#sleep 1
#
## Start the scope daemon
#run scope daemon --filedest localhost:9109 &
#daemon_pid=$!
#sleep 2
#
## Run top
#top -b -d 1 > /dev/null &
#top_pid=$!
#sleep 1
#
## Attach to top
#run scope attach --backtrace --coredump $top_pid
#sleep 1
#
## Crash top
#kill -s SIGSEGV $top_pid
#sleep 5
#
## Check crash and snapshot files exist
#is_file /tmp/appscope/${top_pid}/snapshot_*
#is_file /tmp/appscope/${top_pid}/info_*
#is_file /tmp/appscope/${top_pid}/core_*
#is_file /tmp/appscope/${top_pid}/cfg_*
#is_file /tmp/appscope/${top_pid}/backtrace_*
#
## Check files were received by listener
#wordPresentInFile "snapshot_" "crash.out"
#wordPresentInFile "info_" "crash.out"
#wordPresentInFile "cfg_" "crash.out"
#wordPresentInFile "backtrace_" "crash.out"
#
## Kill scope daemon process
#kill $daemon_pid
#
#endtest


#
# Scope snapshot (same namespace)
#
starttest "Scope snapshot"

top -b -d 1 > /dev/null &
top_pid=$!
sleep 2

SCOPE_SNAPSHOT_COREDUMP=true SCOPE_SNAPSHOT_BACKTRACE=true scope --ldattach $top_pid
returns 0
sleep 2

kill -s SIGSEGV $top_pid
sleep 2

run scope snapshot $top_pid
returns 0
sleep 2

is_file /tmp/appscope/${top_pid}/snapshot_*
is_file /tmp/appscope/${top_pid}/info_*
is_file /tmp/appscope/${top_pid}/core_*
is_file /tmp/appscope/${top_pid}/cfg_*
is_file /tmp/appscope/${top_pid}/backtrace_*

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

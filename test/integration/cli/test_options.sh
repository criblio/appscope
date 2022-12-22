#!/bin/bash
# 
# Cribl AppScope Command-Line Option Tests
#

declare -i ERR=0

ARCH=$(uname -m)

run() {
    CMD="$@"
    echo "  \`${CMD}\`"
    OUT=$(${CMD} 2>&1)
    RET=$?
}

outputs() {
    if ! grep "$1" <<< "$OUT" >/dev/null; then
        echo "    * Expected \"$1\" in output of \`$CMD\`"
        ERR+=1
    fi
}

doesnt_output() {
    if grep "$1" <<< "$OUT" >/dev/null; then
        echo "    * Didn't expect \"$1\" in output of \`$CMD\`"
        ERR+=1
    fi
}

returns() {
    if [ "$RET" != "$1" ]; then
        echo "    * Expected \`$CMD\` to return $1, got $RET"
        ERR+=1
    fi
}

echo "================================="
echo "    Command Line Options Test"
echo "================================="

run ./bin/linux/${ARCH}/scope -n
outputs "missing required value for -n option"
returns 1

run ./bin/linux/${ARCH}/scope -n 1 ls
outputs "error: --namespace option requires --configure/--unconfigure or --service/--unservice option"
returns 1

run ./bin/linux/${ARCH}/scope -c
outputs "missing required value for -c option"
returns 1

run ./bin/linux/${ARCH}/scope -s
outputs "missing required value for -s option"
returns 1

run ./bin/linux/${ARCH}/scope -s dummy_service_value -c dummy_filter_value
outputs "error: --configure/--unconfigure and --service/--unservice cannot be used together"
returns 1

run ./bin/linux/${ARCH}/scope -a dummy_service_value -s 1
outputs "error: --ldattach/--lddetach and --service/--unservice cannot be used together"
returns 1

run ./bin/linux/${ARCH}/scope -c dummy_filter_value -a 1
outputs "error: --ldattach/--lddetach and --configure/--unconfigure cannot be used together"
returns 1

run ./bin/linux/${ARCH}/scope -l 
outputs "missing required value for -l option"
returns 1

run ./bin/linux/${ARCH}/scope -l /does_not_exist echo 
returns 1

run ./bin/linux/${ARCH}/scope --libbasedir /does_not_exist echo 
returns 1

run ./bin/linux/${ARCH}/scope -f /does_not_exist echo 
returns 1

run ./bin/linux/${ARCH}/scope -a 
outputs "missing required value for -a option"
returns 1

run ./bin/linux/${ARCH}/scope -a not_a_pid
outputs "invalid --ldattach PID: not_a_pid"
returns 1

run ./bin/linux/${ARCH}/scope -a -999
outputs "invalid --ldattach PID: -999"
returns 1

run ./bin/linux/${ARCH}/scope -a 999999999
outputs "error: --ldattach, --lddetach PID not a current process"
returns 1

run ./bin/linux/${ARCH}/scope echo foo
outputs foo
returns 0

export SCOPE_LIB_PATH=./lib/linux/${ARCH}/libscope.so
run ./bin/linux/${ARCH}/scope echo
returns 0
export -n SCOPE_LIB_PATH

run ./bin/linux/${ARCH}/scope --ldattach
outputs "missing required value for -a option"
returns 1

run ./bin/linux/${ARCH}/scope -a
outputs "missing required value for -a option"
returns 1

export SCOPE_LIB_PATH=./lib/linux/${ARCH}/libscope.so
run ./bin/linux/${ARCH}/scope -a 999999999
outputs "error: --ldattach, --lddetach PID not a current process: 999999999"
returns 1
export -n SCOPE_LIB_PATH

if [ $ERR -eq "0" ]; then
    echo "Success"
else
    echo "Test Failed"
fi

exit ${ERR}

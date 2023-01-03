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

### Constructor Option Handling ###

run ./bin/linux/${ARCH}/scope -a 
outputs "error: missing required value for -a option"
returns 1

run ./bin/linux/${ARCH}/scope --ldattach
outputs "error: missing required value for --ldattach option"
returns 1

run ./bin/linux/${ARCH}/scope -d 
outputs "error: missing required value for -d option"
returns 1

run ./bin/linux/${ARCH}/scope --lddetach
outputs "error: missing required value for --lddetach option"
returns 1

run ./bin/linux/${ARCH}/scope -n
outputs "error: missing required value for -n option"
returns 1

run ./bin/linux/${ARCH}/scope --namespace
outputs "error: missing required value for --namespace option"
returns 1

run ./bin/linux/${ARCH}/scope -c
outputs "error: missing required value for -c option"
returns 1

run ./bin/linux/${ARCH}/scope --configure
outputs "error: missing required value for --configure option"
returns 1

run ./bin/linux/${ARCH}/scope -s
outputs "error: missing required value for -s option"
returns 1

run ./bin/linux/${ARCH}/scope --service
outputs "error: missing required value for --service option"
returns 1

run ./bin/linux/${ARCH}/scope -l 
outputs "error: missing required value for -l option"
returns 1

run ./bin/linux/${ARCH}/scope --libbasedir
outputs "error: missing required value for --libbasedir option"
returns 1

run ./bin/linux/${ARCH}/scope -p 
outputs "error: missing required value for -p option"
returns 1

run ./bin/linux/${ARCH}/scope --patch
outputs "error: missing required value for --patch option"
returns 1

run ./bin/linux/${ARCH}/scope -z 
outputs "could not find or execute command"
returns 1

run ./bin/linux/${ARCH}/scope --passthrough
outputs "could not find or execute command"
returns 1

run ./bin/linux/${ARCH}/scope -z ps
outputs "PID"
returns 0

run ./bin/linux/${ARCH}/scope -z ps -ef
outputs "UID"
returns 0

run ./bin/linux/${ARCH}/scope -z -- ps
outputs "PID"
returns 0

run ./bin/linux/${ARCH}/scope -z -- ps -ef
outputs "UID"
returns 0

run ./bin/linux/${ARCH}/scope -a -d
outputs "error: --ldattach and --lddetach cannot be used together"
returns 1

run ./bin/linux/${ARCH}/scope -s -v
outputs "error: --service and --unservice cannot be used together"
returns 1

run ./bin/linux/${ARCH}/scope -c -w
outputs "error: --configure and --unconfigure cannot be used together"
returns 1

run ./bin/linux/${ARCH}/scope -p -a
outputs "error: --passthrough cannot be used with --ldattach/--lddetach or --namespace or --service/--unservice or --configure/--unconfigure"
returns 1

run ./bin/linux/${ARCH}/scope -n 1 ls
outputs "error: --namespace option requires --configure/--unconfigure or --service/--unservice option"
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

run ./bin/linux/${ARCH}/scope -l /does_not_exist echo 
returns 1

run ./bin/linux/${ARCH}/scope --libbasedir /does_not_exist echo 
returns 1

run ./bin/linux/${ARCH}/scope -f /does_not_exist echo 
returns 1

run ./bin/linux/${ARCH}/scope -a not_a_pid
outputs "invalid --ldattach PID: not_a_pid"
returns 1

run ./bin/linux/${arch}/scope -d not_a_pid
outputs "invalid --lddetach PID: not_a_pid"
returns 1

run ./bin/linux/${arch}/scope -n not_a_pid
outputs "invalid --namespace PID: not_a_pid"
returns 1

run ./bin/linux/${ARCH}/scope -a -999
outputs "invalid --ldattach PID: -999"
returns 1

run ./bin/linux/${arch}/scope -d -999
outputs "invalid --lddetach PID: -999"
returns 1

run ./bin/linux/${ARCH}/scope -n -999
outputs "invalid --namespace PID: -999"
returns 1

run ./bin/linux/${ARCH}/scope -a 999999999
outputs "error: --ldattach, --lddetach PID not a current process"
returns 1

export SCOPE_LIB_PATH=./lib/linux/${ARCH}/libscope.so
run ./bin/linux/${ARCH}/scope echo
returns 0
export -n SCOPE_LIB_PATH

export SCOPE_LIB_PATH=./lib/linux/${ARCH}/libscope.so
run ./bin/linux/${ARCH}/scope -a 999999999
outputs "error: --ldattach, --lddetach PID not a current process: 999999999"
returns 1
export -n SCOPE_LIB_PATH



### Main Option Handling ###

run ./bin/linux/${ARCH}/scope echo foo
outputs foo
returns 0

run ./bin/linux/${ARCH}/scope run -- echo foo
outputs foo
returns 0

run ./bin/linux/${ARCH}/scope run -- ps -ef // doesn't work without the '--' (-ef parsed by cli instead) and never did
outputs UID
returns 0

run ./bin/linux/${ARCH}/scope run -a some_auth_token -- echo foo
outputs foo
returns 0

run ./bin/linux/${ARCH}/scope 
outputs Cribl AppScope Command Line Interface
returns 0

run ./bin/linux/${ARCH}/scope -h
outputs Cribl AppScope Command Line Interface
returns 0

run ./bin/linux/${ARCH}/scope --help
outputs Cribl AppScope Command Line Interface
returns 0

run ./bin/linux/${ARCH}/scope logs -h
outputs Displays internal AppScope logs for troubleshooting AppScope itself.
returns 0

run ./bin/linux/${ARCH}/scope logs --help
outputs Displays internal AppScope logs for troubleshooting AppScope itself.
returns 0

run ./bin/linux/${ARCH}/scope run
outputs Usage:
returns 1

run ./bin/linux/${ARCH}/scope attach
outputs Usage:
returns 1

run ./bin/linux/${ARCH}/scope run -a
outputs error: missing required value for -a option
returns 1





if [ $ERR -eq "0" ]; then
    echo "Success"
else
    echo "Test Failed"
fi

exit ${ERR}

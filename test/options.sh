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

run ./bin/linux/${ARCH}/ldscope
outputs "error: missing --attach, --detach, --configure, --unconfigure, --service, --unservice option or EXECUTABLE argument"
outputs "Cribl AppScope"
returns 1

run ./bin/linux/${ARCH}/ldscope -n
outputs "missing required value for -n option"
returns 1

run ./bin/linux/${ARCH}/ldscope -n 1 ls
outputs "error: --namespace option required --configure/--unconfigure or --service/--unservice option"
outputs "Cribl AppScope"
returns 1

run ./bin/linux/${ARCH}/ldscope -c
outputs "missing required value for -c option"
returns 1

run ./bin/linux/${ARCH}/ldscope -s
outputs "missing required value for -s option"
returns 1

run ./bin/linux/${ARCH}/ldscope -s dummy_service_value -c dummy_filter_value
outputs "error: --configure/--unconfigure and --service/--unservice cannot be used together"
outputs "Cribl AppScope"
returns 1

run ./bin/linux/${ARCH}/ldscope -a dummy_service_value -s 1
outputs "error: --attach/--detach and --service/--unservice cannot be used together"
outputs "Cribl AppScope"
returns 1

run ./bin/linux/${ARCH}/ldscope -c dummy_filter_value -a 1
outputs "error: --attach/--detach and --configure/--unconfigure cannot be used together"
outputs "Cribl AppScope"
returns 1

run ./bin/linux/${ARCH}/ldscope -z 
outputs "invalid option: -z"
returns 1

run ./bin/linux/${ARCH}/ldscope -u
doesnt_output "error:"
outputs "Cribl AppScope"
returns 0

run ./bin/linux/${ARCH}/ldscope --usage
doesnt_output "error:"
outputs "Cribl AppScope"
returns 0

run ./bin/linux/${ARCH}/ldscope -h
doesnt_output "error:"
outputs "Cribl AppScope"
outputs "OVERVIEW:"
outputs "CONFIGURATION:"
outputs "METRICS:"
outputs "EVENTS:"
outputs "PROTOCOL DETECTION:"
outputs "PAYLOAD EXTRACTION:"
returns 0

run ./bin/linux/${ARCH}/ldscope -h all
doesnt_output "error:"
outputs "Cribl AppScope"
outputs "OVERVIEW:"
outputs "CONFIGURATION:"
outputs "METRICS:"
outputs "EVENTS:"
outputs "PROTOCOL DETECTION:"
outputs "PAYLOAD EXTRACTION:"
returns 0

run ./bin/linux/${ARCH}/ldscope -h AlL
doesnt_output "error:"
outputs "Cribl AppScope"
outputs "OVERVIEW:"
outputs "CONFIGURATION:"
outputs "METRICS:"
outputs "EVENTS:"
outputs "PROTOCOL DETECTION:"
outputs "PAYLOAD EXTRACTION:"
returns 0

run ./bin/linux/${ARCH}/ldscope -h OvErViEw
doesnt_output "error:"
outputs "Cribl AppScope"
outputs "OVERVIEW:"
doesnt_output "CONFIGURATION:"
doesnt_output "METRICS:"
doesnt_output "EVENTS:"
doesnt_output "PROTOCOL DETECTION:"
doesnt_output "PAYLOAD EXTRACTION:"
returns 0

run ./bin/linux/${ARCH}/ldscope -h bogus
outputs "error: invalid help section"
outputs "Cribl AppScope"
returns 1

run ./bin/linux/${ARCH}/ldscope -l 
outputs "missing required value for -l option"
returns 1

run ./bin/linux/${ARCH}/ldscope -l /does_not_exist echo 
returns 1

run ./bin/linux/${ARCH}/ldscope --libbasedir /does_not_exist echo 
returns 1

run ./bin/linux/${ARCH}/ldscope -f /does_not_exist echo 
returns 1

run ./bin/linux/${ARCH}/ldscope -a 
outputs "missing required value for -a option"
returns 1

if [ "0" == "$(id -u)" ]; then

    run ./bin/linux/${ARCH}/ldscope -a not_a_pid
    outputs "invalid --attach, --detach PID"
    returns 1

    run ./bin/linux/${ARCH}/ldscope -a -999
    outputs "invalid --attach, --detach PID"
    returns 1

    run ./bin/linux/${ARCH}/ldscope -a 999999999
    outputs "error: --attach, --detach PID not a current process"
    returns 1

else 

    # we don't require root unless the pid exists and libscope is not present in the maps file
    run ./bin/linux/${ARCH}/ldscope -a 999999999
    outputs "error: --attach, --detach PID not a current process"
    returns 1

fi

run ./bin/linux/${ARCH}/ldscope echo foo
outputs foo
returns 0

run ./bin/linux/${ARCH}/ldscopedyn
outputs "missing --attach, --detach or EXECUTABLE"
returns 1

run ./bin/linux/${ARCH}/ldscopedyn -z 
outputs "invalid option: -z"
returns 1

run ./bin/linux/${ARCH}/ldscopedyn echo
outputs "SCOPE_LIB_PATH must be set"
returns 1

export SCOPE_LIB_PATH=bogus
run ./bin/linux/${ARCH}/ldscopedyn echo
outputs "library bogus is missing"
returns 1
export -n SCOPE_LIB_PATH

export SCOPE_LIB_PATH=./lib/linux/${ARCH}/libscope.so
run ./bin/linux/${ARCH}/ldscopedyn echo
returns 0
export -n SCOPE_LIB_PATH

run ./bin/linux/${ARCH}/ldscopedyn --attach
outputs "missing value for -a option"
returns 1

run ./bin/linux/${ARCH}/ldscopedyn -a
outputs "missing value for -a option"
returns 1

export SCOPE_LIB_PATH=./lib/linux/${ARCH}/libscope.so
run ./bin/linux/${ARCH}/ldscopedyn -a 999999999
outputs "error: can't get path to executable for pid 999999999"
returns 1
export -n SCOPE_LIB_PATH

export SCOPE_LIB_PATH=./lib/linux/${ARCH}/libscope.so
run ./bin/linux/${ARCH}/ldscopedyn -a 999999999 echo
outputs "ignoring EXECUTABLE argument with --attach, --detach option"
returns 1
export -n SCOPE_LIB_PATH

if [ $ERR -eq "0" ]; then
    echo "Success"
else
    echo "Test Failed"
fi

exit ${ERR}

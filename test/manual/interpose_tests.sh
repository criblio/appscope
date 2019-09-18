#! /bin/bash

LTP=$HOME/ltp/testcases/kernel/
LIBWRAP=$HOME/scope/lib/linux/libwrap.so

run_test() {
if PATH=$PATH:$LTP/$1 LD_PRELOAD=$LIBWRAP $LTP/$1/$2
then
    echo "PASS"
else
    echo "FAIL"
    exit
fi
}

run_test "syscalls/faccessat" "faccessat01"
run_test "syscalls/access" "access01"
run_test "syscalls/access" "access02"
run_test "syscalls/access" "access03"
run_test "fs/stream" "stream01"
run_test "fs/stream" "stream02"
run_test "fs/stream" "stream03"
run_test "fs/stream" "stream04"

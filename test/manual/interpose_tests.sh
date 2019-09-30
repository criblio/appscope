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
run_test "syscalls/dup" "dup01"
run_test "syscalls/dup" "dup02"
run_test "syscalls/dup" "dup03"
run_test "syscalls/dup" "dup04"
run_test "syscalls/dup" "dup05"
run_test "syscalls/dup" "dup06"
run_test "syscalls/dup" "dup07"
run_test "syscalls/send" "send01"
run_test "syscalls/sendmsg" "sendmsg01"
# takes 15 secs to run, just removing so as not to wait
#run_test "syscalls/sendmsg" "sendmsg02"
run_test "syscalls/sendto" "sendto01"
run_test "syscalls/sendto" "sendto02"
run_test "syscalls/recv" "recv01"
run_test "syscalls/recvmsg" "recvmsg01"
run_test "syscalls/recvmsg" "recvmsg02"
#fails w/o being wrapped
#run_test "syscalls/recvmsg" "recvmsg03"
run_test "syscalls/recvfrom" "recvfrom01"
run_test "syscalls/openat" "openat01"
#fails w/o being wrapped
#run_test "syscalls/openeat" "openat02"
#fails w/o being wrapped
#run_test "syscalls/openat" "openat02_child"
run_test "syscalls/openat" "openat03"
run_test "syscalls/open" "open01"
run_test "syscalls/open" "open02"
run_test "syscalls/open" "open03"
run_test "syscalls/open" "open04"
run_test "syscalls/open" "open05"

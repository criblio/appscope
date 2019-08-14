#!/bin/bash

CWD="$(pwd)"

if uname -s 2> /dev/null | grep -i "linux" > /dev/null; then
    OS="linux"
    export LD_LIBRARY_PATH=contrib/cmocka/build/src/
elif uname -s 2> /dev/null | grep -i darwin > /dev/null; then
    OS="macOS"
    export DYLD_LIBRARY_PATH=contrib/cmocka/build/src/
else
    OS="unknown"
fi

# We want to run all tests, even if some of them return errors.
# Then after all tests are run, we want to report (via the return code)
# if any errors occurred.  ERR maintains this state.
declare -i ERR=0

test/${OS}/cfgutilstest; ERR+=$?
test/${OS}/cfgtest; ERR+=$?
test/${OS}/transporttest; ERR+=$?
test/${OS}/logtest; ERR+=$?
test/${OS}/outtest; ERR+=$?
test/${OS}/formattest; ERR+=$?
test/${OS}/dbgtest; ERR+=$?
if [ "${OS}" = "linux" ]; then
    test/${OS}/glibcvertest; ERR+=$?
fi





#                ^
#                |
#                |
# add new tests here #


# wraptest has special requirements, env wise...
export SCOPE_HOME=${CWD}/test/
if [ "${OS}" = "linux" ]; then
    export LD_LIBRARY_PATH=lib/${OS}:${LD_LIBRARY_PATH}
elif [ "${OS}" = "macOS" ]; then
    export DYLD_LIBRARY_PATH=lib/${OS}:${DYLD_LIBRARY_PATH}
    export DYLD_INSERT_LIBRARIES=${CWD}/lib/${OS}/libwrap.so
    export DYLD_FORCE_FLAT_NAMESPACE=1
fi
test/${OS}/wraptest; ERR+=$?


# I think this is unnecessary, but...
unset SCOPE_HOME
if [ "${OS}" = "linux" ]; then
    unset LD_LIBRARY_PATH
elif [ "${OS}" = "macOS" ]; then
    unset DYLD_LIBRARY_PATH
    unset DYLD_INSERT_LIBRARIES
    unset DYLD_FORCE_FLAT_NAMESPACE
fi

exit ${ERR}

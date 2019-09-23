#!/bin/bash

accumulate_coverage() {
    # This function accumulates coverage info (gcda files) from test
    # to test.  It merges new coverage info (current dir) into coverage
    # info from previous tests (coverage.tmp dir).


    # the first time this runs, coverage.tmp won't exist.  Create it.
    if [ ! -d coverage.tmp ]; then
        gcov-tool rewrite . -o coverage.tmp
        rm *gcda
        return 0
    fi

    # on subsequent iterations, merge with coverage.tmp
    gcov-tool merge . coverage.tmp -o coverage.merged
    rm *gcda

    # move coverage.merged to coverage.tmp
    rm -rf coverage.tmp
    gcov-tool rewrite coverage.merged -o coverage.tmp
    rm -rf coverage.merged
    return 0
}

return_coverage() {
    # This function moves accumlated coverage info
    # (from coverage.tmp dir) into the current dir
    gcov-tool rewrite coverage.tmp -o .
    rm -rf coverage.tmp

}

run_test() {


    # print out instructions for how to run one at a time...
    if [ ! -z "${LD_LIBRARY_PATH}" ]; then
        echo "Running LD_LIBRARY_PATH=$LD_LIBRARY_PATH $1"
    fi
    if [ ! -z "${DYLD_LIBRARY_PATH}" ]; then
        echo "Running DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH $1"
    fi

    # run the test
    $1

    # accumulate errors reported by the return value of the test
    ERR+=$?

    # accumulate coverage information
    accumulate_coverage
}

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

run_test test/${OS}/cfgutilstest
run_test test/${OS}/cfgtest
run_test test/${OS}/transporttest
run_test test/${OS}/logtest
run_test test/${OS}/outtest
run_test test/${OS}/formattest
run_test test/${OS}/dbgtest
if [ "${OS}" = "linux" ]; then
    run_test test/${OS}/glibcvertest
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
run_test test/${OS}/wraptest


return_coverage


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

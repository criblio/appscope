#!/bin/bash

NUM=0
INFOLIST=""
ENVVARS=""

accumulate_coverage() {
    # This function accumulates coverage info (gcda files) from test
    # to test.  It merges new coverage info (current dir) into coverage
    # info from previous tests (coverage.tmp dir).

    FILE=coverage/coverage$NUM.info
    lcov --capture --directory . --output-file $FILE
    INFOLIST=$INFOLIST"$FILE "
    rm -f *\.gcda
    ((NUM++))

    return 0
}

report_final_coverage() {

    genhtml -o coverage $INFOLIST
    rm -f *\.gcno *\.o *\.gcda
}

run_test() {

    echo "$ENVVARS$1"

    # run the test
    (export $ENVVARS; $1 2>&1)
    #(export $ENVVARS; valgrind $1)

    # accumulate errors reported by the return value of the test
    ERR+=$?

    # accumulate coverage information
    accumulate_coverage
}

CWD="$(pwd)"

if [ -d $CWD/coverage ]; then
    rm -rf $CWD/coverage
fi
mkdir $CWD/coverage

if uname -s 2> /dev/null | grep -i "linux" > /dev/null; then
    OS="linux"
    ENVVARS=$ENVVARS"LD_LIBRARY_PATH=contrib/build/cmocka/src/ "
elif uname -s 2> /dev/null | grep -i darwin > /dev/null; then
    OS="macOS"
    ENVVARS=$ENVVARS"DYLD_LIBRARY_PATH=contrib/build/cmocka/src/ "
else
    OS="unknown"
fi


# We want to run all tests, even if some of them return errors.
# Then after all tests are run, we want to report (via the return code)
# if any errors occurred.  ERR maintains this state.
declare -i ERR=0

run_test test/${OS}/vdsotest
run_test test/${OS}/strsettest
run_test test/${OS}/cfgutilstest
run_test test/${OS}/cfgtest
run_test test/${OS}/transporttest
run_test test/${OS}/logtest
run_test test/${OS}/utilstest
run_test test/${OS}/mtctest
run_test test/${OS}/evtformattest
run_test test/${OS}/ctltest
run_test test/${OS}/mtcformattest
run_test test/${OS}/circbuftest
run_test test/${OS}/linklisttest
run_test test/${OS}/comtest
run_test test/${OS}/dbgtest
run_test test/${OS}/searchtest
run_test test/${OS}/httpstatetest
if [ "${OS}" = "linux" ]; then
    run_test test/${OS}/glibcvertest
    run_test test/${OS}/reporttest
    run_test test/${OS}/javabcitest
    run_test test/${OS}/httpheadertest
fi
run_test test/${OS}/httpaggtest
run_test test/${OS}/selfinterposetest

if [ "${OS}" = "linux" ]; then
    SAVEVARS=$ENVARS
    ENVVARS=$ENVVARS"LD_PRELOAD=./lib/linux/$(uname -m)/libscope.so ""SCOPE_FILTER=false ""SCOPE_CRIBL_ENABLE=false ""SCOPE_METRIC_DEST=file:///tmp/dnstest.log ""SCOPE_METRIC_VERBOSITY=9 ""SCOPE_SUMMARY_PERIOD=1 "
    run_test test/${OS}/dnstest
    ENVARS=$SAVEVARS
    rm -f "/tmp/dnstest.log"

    test/access_rights.sh 2>&1
    ERR+=$?

    test/unixpeer.sh 2>&1
    ERR+=$?

    test/undefined_sym.sh 2>&1
    ERR+=$?
fi

test/options.sh 2>&1
ERR+=$?

#                ^
#                |
#                |
# add new tests here #


# wraptest has special requirements, env wise...
#ENVVARS="SCOPE_HOME=${CWD}/test/ "
#if [ "${OS}" = "linux" ]; then
#    ENVVARS=$ENVVARS"LD_LIBRARY_PATH=lib/${OS}:contrib/build/cmocka/src/ "
#elif [ "${OS}" = "macOS" ]; then
#    ENVVARS=$ENVVARS"DYLD_LIBRARY_PATH=lib/${OS}:contrib/build/cmocka/src/ "
#    ENVVARS=$ENVVARS"DYLD_INSERT_LIBRARIES=${CWD}/lib/${OS}/libscope.so "
#    ENVVARS=$ENVVARS"DYLD_FORCE_FLAT_NAMESPACE=1 "
#fi
#run_test test/${OS}/wraptest


report_final_coverage


exit ${ERR}

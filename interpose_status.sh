#!/bin/bash

LIB_NAME=libscope.so


# Setup some things based on platform
if uname -s 2> /dev/null | grep -i "linux" > /dev/null; then
    PRELOAD=LD_PRELOAD
    REL_PATH=lib/linux/${LIB_NAME}
    NM_OPTS=-D
elif uname -s 2> /dev/null | grep -i "darwin" > /dev/null; then
    PRELOAD=DYLD_INSERT_LIBRARIES
    REL_PATH=lib/macOS/${LIB_NAME}
    NM_OPTS=-a
else
    echo "ERROR: Undetermined platform.  Exiting..." >&2
    exit 1
fi


# We need to resolve the scope library to see what it interposes.
# In priority order, use:
#  1) LD_PRELOAD / DYLD_INSERT_LIBRARIES env variables
#  2) SCOPE_HOME if defined
#  3) find from current directory
if [[ ${LD_PRELOAD} == *"$LIB_NAME"* ]]; then
    LIB_PATH=${LD_PRELOAD}
elif [[ ${DYLD_INSERT_LIBRARIES} == *"$LIB_NAME"* ]]; then
    LIB_PATH=${DYLD_INSERT_LIBRARIES}
elif [ ! -z "$SCOPE_HOME" ]; then
    if [ -f "${SCOPE_HOME}/${REL_PATH}" ]; then
        LIB_PATH="${SCOPE_HOME}/${REL_PATH}"
    fi
else
    LIB_FIND=`find . -type f -name "$LIB_NAME" | grep -v dSYM | head -n1`
    if [[ ${LIB_FIND} == *"$LIB_NAME"* ]]; then
        LIB_PATH=${LIB_FIND}
    fi
fi

if [ -z $LIB_PATH ]; then
    echo "ERROR." >&2
    echo "Couldn't find $LIB_NAME which is required for $0 to give helpful feedback." >&2
    echo "Please rerun with one of the following changes:" >&2
    echo " o) set ${PRELOAD} env variable with path to $LIB_NAME" >&2
    echo " o) set SCOPE_HOME env variable to a directory which could resolve ${REL_PATH}" >&2
    echo " o) run this script from a directory which contains or is a parent of $LIB_NAME" >&2
    exit 1
fi

# We need to get a command as an argument
if [ -z "$1" ]; then
    echo "ERROR: $0 requires a command as an argument." >&2
    echo "  e.g. $0 /bin/ps" >&2
    exit 1
fi

if [ -f "$1" ]; then
    CMD=$1
else
    CMD=`which $1`
fi

if [ -z "$CMD" ]; then
    echo "ERROR: Could not resolve $1 as a command.  Try specifying an absolute path." >&2
    exit 1
fi


# We should be good to go!
echo "Processing ${CMD} with scope library ${LIB_PATH}"


INTERPOSED_FNS=`nm ${NM_OPTS} ${LIB_PATH} | grep " T " | cut -c20-100 | grep -vE "(_init)|(_fini)"`
#echo "${INTERPOSED_FNS}"

nm ${NM_OPTS} $CMD | grep -E " [WU] " | cut -c20-100 > ./nm.out
NM_FUNCTIONS_NUM=`cat ./nm.out | wc -l`

SCOPE_NUM=0
SCOPE_ARRAY=()
for FN in ${INTERPOSED_FNS[*]}; do
    GREP_RESULT=`grep "^${FN}$" ./nm.out`
    if [ $? == 0 ]; then
        SCOPE_ARRAY+="${GREP_RESULT}\n"
        ((SCOPE_NUM+=1))
    fi
done


echo "Found $NM_FUNCTIONS_NUM dynamically linked functions in $CMD"
echo "Of these scope will interpose these $SCOPE_NUM functions:"
echo -e "$SCOPE_ARRAY" | sort | tail -n +2 | sed 's/^/    /'

rm ./nm.out

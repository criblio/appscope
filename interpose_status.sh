#!/bin/bash

LIB_NAME=libwrap.so
# We need to resolve the scope library to see what it interposes.
# In priority order, use:
#  1) LD_PRELOAD / DYLD_INSERT_LIBRARIES env variables
#  2) CRIBL_HOME if defined
#  3) find from current directory
if [[ ${LD_PRELOAD} == *"$LIB_NAME"* ]]; then
    LIB_PATH=${LD_PRELOAD}
elif [[ ${DYLD_INSERT_LIBRARIES} == *"$LIB_NAME"* ]]; then
    LIB_PATH=${DYLD_INSERT_LIBRARIES}
elif [ ! -z "$CRIBL_HOME" ]; then
    if [ -f "${CRIBL_HOME}/lib/linux/$LIB_NAME" ]; then
        LIB_PATH="${CRIBL_HOME}/lib/linux/$LIB_NAME"
    elif [ -f "$CRIBL_HOME/lib/macOS/$LIB_NAME" ]; then
        LIB_PATH="$CRIBL_HOME/lib/macOS/$LIB_NAME"
    fi
else
    LIB_FIND=`find . -type f -name "$LIB_NAME" | grep -v dSYM | head -n1`
    if [[ ${LIB_FIND} == *"$LIB_NAME"* ]]; then
        LIB_PATH=${LIB_FIND}
    fi
fi

if [ -z $LIB_PATH ]; then
    echo "Couldn't find $LIB_NAME which is required for $0 to give helpful feedback." >&2
    echo "Please rerun with one of the following changes:" >&2
    echo " o) set LD_PRELOAD env variable with path to $LIB_NAME (assuming linux)" >&2
    echo " o) set DYLD_INSERT_LIBRARIES env variable with path to $LIB_NAME (assuming mac)" >&2
    echo " o) set CRIBL_HOME env variable to a directory which could resolve " >&2
    echo "        lib/linux/$LIB_NAME (linux) or lib/macOS/$LIB_NAME (mac)" >&2
    echo " o) run this script from a directory which contains or is a parent of $LIB_NAME" >&2
    exit 1
fi

if [ -z "$1" ]; then
    echo "$0 requires a command as an argument." >&2
    echo "  e.g. $0 /bin/ps" >&2
    exit 1
fi

if [ -f "$1" ]; then
    CMD=$1
else
    CMD=`which $1`
fi

if [ -z "$CMD" ]; then
    echo "Could not resolve $1 as a command.  Try specifying an absolute path." >&2
    exit 1
fi

echo "Processing ${CMD}"


INTERPOSED_FNS=`nm -a ${LIB_PATH} | grep " T " | cut -c20-100 | grep -vE "(_init)|(_fini)"`
#echo "${INTERPOSED_FNS}"

nm -a $CMD | grep -E " [WU] " | cut -c20-100 >> ./nm.out
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

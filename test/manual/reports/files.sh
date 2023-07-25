#! /bin/bash

SCOPE_PATH="."
RESULT="./file_results.txt"
cmd=""
id=""
dir=0

while test $# -gt 0
do
    case "$1" in
        --cmd) cmd="$2"
               arg1="$3"
               arg2="$4"
               echo "executing $cmd"
               ;;
        --dirs) dir=1
               echo "export directory names only"
               ;;
        --raw) raw=$2
               echo "using the events file at $2."
               ;;
        --id) id="--id $2"
               echo "using the events file defined by id $2."
               ;;
        --*) echo "bad option $1"
               ;;
    esac
    shift
done

if [[ $cmd != "" ]]; then
    $SCOPE_PATH/scope run -- $cmd $arg1 $arg2
fi

if [[ $raw != "" ]]; then
    if [ $dir -ne 0  ]; then
        grep fs.open $raw | jq '.body.data | "\(.file)"' | xargs dirname | sort | uniq > $RESULT
    else
        grep fs.open $raw | jq '.body.data | "\(.file)"' | sort | uniq > $RESULT
    fi
elif [ $dir -ne 0  ]; then
    $SCOPE_PATH/scope events -aj $id | grep '"fs.open"' | jq -r '.data.file' | xargs dirname | sort | uniq  > $RESULT
else
    $SCOPE_PATH/scope events -aj $id | grep '"fs.open"' | jq -r '.data.file' | sort | uniq > $RESULT
fi

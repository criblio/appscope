#! /bin/bash

SCOPE_PATH="."
RESULT="./connections_results.txt"
cmd=""

while test $# -gt 0
do
    case "$1" in
        --cmd) cmd="$2"
               arg1="$3"
               arg2="$4"
               echo "executing $cmd"
               ;;
        --raw) raw=$2
               echo "using the events file at $2."
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
    grep net.open $raw | grep net.open ~/.scope/history/npm_82_24392_1690230424144945243/events.json | jq '.body.data | "\(.net_host_ip):\(.net_host_port) \(.net_peer_ip):\(.net_peer_port)"' | sort | uniq > $RESULT
else
    $SCOPE_PATH/scope events -aj | grep '"net.open"' | jq '.data | "\(.net_host_ip) \(.net_host_port) \(.net_peer_ip) \(.net_peer_port) "' | sort | uniq > $RESULT
fi

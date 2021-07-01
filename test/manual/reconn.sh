#! /bin/bash

num=1
pkill -f tcpserver

while [ $num -le 10 ]
do
    touch /tmp/reconn.log
      
    ~/scope/utils/tcpserver 9109 | grep --line-buffered '{"format":' > /tmp/reconn.log &

    sleep 1
    wc -l /tmp/reconn.log

    grep -q '"format":"ndjson"' /tmp/reconn.log
    if [ $? -ne 0 ]; then
        echo "error: missing ndjson"
        pkill -f tcpserver
        rm /tmp/reconn.log
        exit 1
    fi

    grep -q '"format":"scope"' /tmp/reconn.log
    if [ $? -ne 0 ]; then
        echo "error: missing scope"
        pkill -f tcpserver
        rm /tmp/reconn.log
        exit 1
    fi

    pkill -f tcpserver
    rm /tmp/reconn.log
    num=$((num+1))
done

pkill -f tcpserver
echo "Success"
exit 0

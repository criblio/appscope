#! /bin/bash

OUTFILE="samples.evt"

SYSCALLFILE="syscalls.evt"
EVTLIST="fs.open fs.close fs.delete net.error fs.error fs.stat fs.open fs.close fs.duration net.duration fs.read fs.write fs.seek fs.stat net.port net.tcp net.udp net.other net.rx net.tx net.open net.close"

HTTPFILE="http_client.evt"
HTTPLIST="http.req http.resp dns.req dns.resp net.app console"

HTTPSERVERFILE="http_server.evt"
HTTPSERVERLIST="http.req http.resp file"

getEvent()
{
    echo "using $1"
    for evt in $2; do
        #echo $evt
        grep -m 1 $evt $1 >> $OUTFILE

        if [ $? -ne "0" ]; then
            echo "Missing $evt"
        fi
    done
}

rm -f $OUTFILE
getEvent "$SYSCALLFILE" "$EVTLIST"
getEvent "$HTTPFILE" "$HTTPLIST"
getEvent "$HTTPSERVERFILE" "$HTTPSERVERLIST"

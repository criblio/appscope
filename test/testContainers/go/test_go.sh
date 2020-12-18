#! /bin/bash

DEBUG=0  # set this to 1 to capture the EVT_FILE for each test

FAILED_TEST_LIST=""
FAILED_TEST_COUNT=0

EVT_FILE="/go/events.log"
touch $EVT_FILE

starttest(){
    CURRENT_TEST=$1
    echo "==============================================="
    echo "             Testing $CURRENT_TEST             "
    echo "==============================================="
    ERR=0
}

evaltest(){
    echo "             Evaluating $CURRENT_TEST"
}

endtest(){
    if [ $ERR -eq "0" ]; then
        RESULT=PASSED
    else
        RESULT=FAILED
        FAILED_TEST_LIST+=$CURRENT_TEST
        FAILED_TEST_LIST+=" "
        FAILED_TEST_COUNT=$(($FAILED_TEST_COUNT + 1))
    fi

    echo "*************** $CURRENT_TEST $RESULT ***************"
    echo ""
    echo ""

    # copy the EVT_FILE to help with debugging
    if (( $DEBUG )) || [ $RESULT == "FAILED" ]; then
        cp $EVT_FILE $EVT_FILE.$CURRENT_TEST
    fi

    rm $EVT_FILE
}


#
# plainServerDynamic
#
starttest plainServerDynamic
cd /go/net
PORT=80
scope ./plainServerDynamic ${PORT} &

# this sleep gives the server a chance to bind to the port
# before we try to hit it with curl
sleep 0.5
curl http://localhost:${PORT}/hello
ERR+=$?

# This stops plainServerDynamic
pkill -f plainServerDynamic

# this sleep gives plainServerDynamic a chance to report its events on exit
sleep 0.5

evaltest

grep plainServerDynamic $EVT_FILE | grep http-req > /dev/null
ERR+=$?
grep plainServerDynamic $EVT_FILE | grep http-resp > /dev/null
ERR+=$?
grep plainServerDynamic $EVT_FILE | grep http-metrics > /dev/null
ERR+=$?

endtest


#
# plainServerStatic
#
starttest plainServerStatic
cd /go/net
PORT=81
scope ./plainServerStatic ${PORT} &

# this sleep gives the server a chance to bind to the port
# before we try to hit it with curl
sleep 0.5
curl http://localhost:${PORT}/hello
ERR+=$?

# This stops plainServerStatic
pkill -f plainServerStatic

# this sleep gives plainServerStatic a chance to report its events on exit
sleep 0.5

evaltest

grep plainServerStatic $EVT_FILE | grep http-req > /dev/null
ERR+=$?
grep plainServerStatic $EVT_FILE | grep http-resp > /dev/null
ERR+=$?
grep plainServerStatic $EVT_FILE | grep http-metrics > /dev/null
ERR+=$?

endtest


#
# tlsServerDynamic
#
starttest tlsServerDynamic
cd /go/net
PORT=4430
scope ./tlsServerDynamic ${PORT} &

# this sleep gives the server a chance to bind to the port
# before we try to hit it with curl
sleep 0.5
curl -k --key server.key --cert server.crt https://localhost:${PORT}/hello
ERR+=$?

# This stops tlsServerDynamic
pkill -f tlsServerDynamic

# this sleep gives tlsServerDynamic a chance to report its events on exit
sleep 0.5

evaltest

grep tlsServerDynamic $EVT_FILE | grep http-req > /dev/null
ERR+=$?
grep tlsServerDynamic $EVT_FILE | grep http-resp > /dev/null
ERR+=$?
grep tlsServerDynamic $EVT_FILE | grep http-metrics > /dev/null
ERR+=$?

endtest


#
# tlsServerStatic
#
starttest tlsServerStatic
cd /go/net
PORT=4431
STRUCT_PATH=/go/net/go_offsets.txt
SCOPE_GO_STRUCT_PATH=$STRUCT_PATH scope ./tlsServerStatic ${PORT} &

# this sleep gives the server a chance to bind to the port
# before we try to hit it with curl
sleep 0.5
curl -k --key server.key --cert server.crt https://localhost:${PORT}/hello
ERR+=$?

# This stops tlsServerStatic
pkill -f tlsServerStatic

# this sleep gives tlsServerStatic a chance to report its events on exit
sleep 0.5

evaltest

grep tlsServerStatic $EVT_FILE | grep http-req > /dev/null
ERR+=$?
grep tlsServerStatic $EVT_FILE | grep http-resp > /dev/null
ERR+=$?
grep tlsServerStatic $EVT_FILE | grep http-metrics > /dev/null
ERR+=$?

endtest


#
# plainClientDynamic
#
starttest plainClientDynamic
cd /go/net
scope ./plainClientDynamic
ERR+=$?

# this sleep gives plainClientDynamic a chance to report its events on exit
sleep 0.5

evaltest

grep plainClientDynamic $EVT_FILE | grep http-req > /dev/null
ERR+=$?
grep plainClientDynamic $EVT_FILE | grep http-resp > /dev/null
ERR+=$?
grep plainClientDynamic $EVT_FILE | grep http-metrics > /dev/null
ERR+=$?

endtest


#
# plainClientStatic
#
starttest plainClientStatic
cd /go/net
scope ./plainClientStatic
ERR+=$?

# this sleep gives plainClientStatic a chance to report its events on exit
sleep 0.5

evaltest

grep plainClientStatic $EVT_FILE | grep http-req > /dev/null
ERR+=$?
grep plainClientStatic $EVT_FILE | grep http-resp > /dev/null
ERR+=$?
grep plainClientStatic $EVT_FILE | grep http-metrics > /dev/null
ERR+=$?

endtest


#
# tlsClientDynamic
#
starttest tlsClientDynamic
cd /go/net
scope ./tlsClientDynamic
ERR+=$?

# this sleep gives tlsClientDynamic a chance to report its events on exit
sleep 0.5

evaltest

grep tlsClientDynamic $EVT_FILE | grep http-req > /dev/null
ERR+=$?
grep tlsClientDynamic $EVT_FILE | grep http-resp > /dev/null
ERR+=$?
grep tlsClientDynamic $EVT_FILE | grep http-metrics > /dev/null
ERR+=$?

endtest


#
# tlsClientStatic
#
starttest tlsClientStatic
cd /go/net
SCOPE_GO_STRUCT_PATH=$STRUCT_PATH scope ./tlsClientStatic
ERR+=$?

# this sleep gives tlsClientStatic a chance to report its events on exit
sleep 0.5

evaltest

grep tlsClientStatic $EVT_FILE | grep http-req > /dev/null
ERR+=$?
grep tlsClientStatic $EVT_FILE | grep http-resp > /dev/null
ERR+=$?
grep tlsClientStatic $EVT_FILE | grep http-metrics > /dev/null
ERR+=$?

endtest


#
# test_go_struct_server
#
starttest test_go_struct_server
cd /go

./test_go_struct.sh $STRUCT_PATH /go/net/tlsServerStatic Server
ERR+=$?

evaltest

touch $EVT_FILE
endtest


#
# test_go_struct_client
#
starttest test_go_struct_client
cd /go

./test_go_struct.sh $STRUCT_PATH /go/net/tlsClientStatic Client
ERR+=$?

evaltest

touch $EVT_FILE
endtest


#
# fileThread
#
starttest fileThread
cd /go/thread
scope ./fileThread
ERR+=$?
evaltest

grep fileThread $EVT_FILE > /dev/null
ERR+=$?

endtest


#
# cgoDynamic
#
starttest cgoDynamic
cd /go/cgo
LD_LIBRARY_PATH=. scope ./cgoDynamic
ERR+=$?
evaltest

grep cgoDynamic $EVT_FILE > /dev/null
ERR+=$?

endtest


#
# cgoStatic
#
starttest cgoStatic
cd /go/cgo
scope ./cgoStatic
ERR+=$?
evaltest

grep cgoStatic $EVT_FILE > /dev/null
ERR+=$?

endtest

#
#  influxdb tests
#
dbfile="/go/influx/db/meta/meta.db"
influx_verbose=0

influx_start_server() {
    rm -f /go/influx/db/*.event

    if (( $influx_verbose )); then
        SCOPE_EVENT_DEST=file:///go/influx/db/influxd.event scope $1 &
    else
        SCOPE_EVENT_DEST=file:///go/influx/db/influxd.event scope $1 2>/dev/null &
    fi

    until test -e "$dbfile" ; do
	    sleep 1
    done

}

influx_eval() {
    pkill -f $2

    pexist="influxd"
    until test -z "$pexist" ; do
	    sleep 1
	    pexist=`ps -ef | grep influxd | grep config`
    done

    evaltest
    cnt=`grep -c http-req /go/influx/db/influxd.event`
    if (test "$cnt" -lt $1); then
	    echo "ERROR: Server count is $cnt"
	    ERR+=1
    else
	    echo "Server Success count is $cnt"
    fi

    if [ -e  "/go/influx/db/influxc.event" ]; then
        cnt=`grep -c http-req /go/influx/db/influxc.event`
        if (test "$cnt" -lt $1); then
	        echo "ERROR: Client count is $cnt"
	        ERR+=1
        else
	        echo "Client Success count is $cnt"
        fi
    fi

    rm -r /go/influx/db/*
    touch /go/influx/db/data.txt
    touch $EVT_FILE
    endtest

}

#
# influx static stress test
#
# we've seen that the stress test does not run repeatedly
# in a container/EC2 environment. There is a standalone
# script that will run stress, Commentend out here.
#
sleep 1

# This test has race conditions where it closes sockets
# while other threads are reading/writing data on them.
# SCOPE_HTTP_SERIALIZE_ENABLE ensures that a close and
# read/write can't happen at the same time.
export SCOPE_HTTP_SERIALIZE_ENABLE=true
starttest influx_static_stress

influx_start_server "/go/influx/influxd_stat --config /go/influx/influxdb.conf"

SCOPE_EVENT_DEST=file:///go/influx/db/influxc.event scope /go/influx/stress_test insert -n 1000000 -f

influx_eval 50 scope

unset SCOPE_HTTP_SERIALIZE_ENABLE

#
# influx static TLS test
#
sleep 1
starttest influx_static_tls

influx_start_server "/go/influx/influxd_stat --config /go/influx/influxdb_ssl.conf"

SCOPE_EVENT_DEST=file:///go/influx/db/influxc.event /go/influx/influx_stat -ssl -unsafeSsl -host localhost -execute 'CREATE DATABASE goats'
SCOPE_EVENT_DEST=file:///go/influx/db/influxc.event scope /go/influx/influx_stat -ssl -unsafeSsl -host localhost -execute 'SHOW DATABASES'

if (test $influx_verbose -eq 0); then
    sleep 1
fi

SCOPE_EVENT_DEST=file:///go/influx/db/influxc.event scope /go/influx/influx_stat -ssl -unsafeSsl -host localhost -import -path=/go/influx/data.txt -precision=s
SCOPE_EVENT_DEST=file:///go/influx/db/influxc.event scope /go/influx/influx_stat -ssl -unsafeSsl -host localhost -execute 'SHOW DATABASES'

influx_eval 2 scope


#
# influx dynamic TLS test
#
sleep 1
starttest influx_dynamic_tls

influx_start_server "/go/influx/influxd_dyn --config /go/influx/influxdb_ssl.conf"

SCOPE_EVENT_DEST=file:///go/influx/db/influxc.event scope /go/influx/influx_dyn -ssl -unsafeSsl -host localhost -execute 'CREATE DATABASE goats'
SCOPE_EVENT_DEST=file:///go/influx/db/influxc.event scope /go/influx/influx_dyn -ssl -unsafeSsl -host localhost -execute 'SHOW DATABASES'

influx_eval 2 influxd

#
# influx static clear test
#
sleep 1
starttest influx_static_clear

influx_start_server "/go/influx/influxd_stat --config /go/influx/influxdb.conf"

SCOPE_EVENT_DEST=file:///go/influx/db/influxc.event scope /go/influx/influx_stat -host localhost -execute 'CREATE DATABASE sheep'
SCOPE_EVENT_DEST=file:///go/influx/db/influxc.event scope /go/influx/influx_stat -host localhost -execute 'SHOW DATABASES'

influx_eval 2 scope


#
# influx dynamic clear test
#
sleep 1
starttest influx_dynamic_clear

influx_start_server "/go/influx/influxd_dyn --config /go/influx/influxdb.conf"

SCOPE_EVENT_DEST=file:///go/influx/db/influxc.event scope /go/influx/influx_dyn -host localhost -execute 'CREATE DATABASE sheep'
SCOPE_EVENT_DEST=file:///go/influx/db/influxc.event scope /go/influx/influx_dyn -host localhost -execute 'SHOW DATABASES'

if (test $influx_verbose -eq 0); then
    sleep 1
fi


SCOPE_EVENT_DEST=file:///go/influx/db/influxc.event scope /go/influx/influx_dyn -import -path=/go/influx/data.txt -precision=s
SCOPE_EVENT_DEST=file:///go/influx/db/influxc.event scope /go/influx/influx_dyn -host localhost -execute 'SHOW DATABASES'

influx_eval 2 influxd



#
# Done: print results
#
if (( $FAILED_TEST_COUNT == 0 )); then
    echo ""
    echo ""
    echo "*************** ALL TESTS PASSED ***************"
else
    echo "*************** SOME TESTS FAILED ***************"
    echo "Failed tests: $FAILED_TEST_LIST"
    echo "Refer to these files for more info:"
    for FAILED_TEST in $FAILED_TEST_LIST; do
        echo "  $EVT_FILE.$FAILED_TEST"
    done
fi

exit ${FAILED_TEST_COUNT}

#! /bin/bash

influx_verbose=0
scope_path="../../../../bin/linux/$(uname -m)/scope"
influx_path="./"
dbfile="$influx_path/db/meta/meta.db"

echo "==============================================="
echo "             Testing influx static stress      "
echo "==============================================="

rm -f $influx_path/db/*.event

if (( $influx_verbose )); then
#    SCOPE_LOG_LEVEL=debug SCOPE_EVENT_HTTP=true SCOPE_EVENT_METRIC=true SCOPE_METRIC_ENABLE=false SCOPE_EVENT_DEST=file://$influx_path/db/influxd.event $scope_path $influx_path/influxd_dyn --config $influx_path/stress_local.conf &
    SCOPE_LOG_LEVEL=debug SCOPE_EVENT_HTTP=true SCOPE_EVENT_METRIC=true SCOPE_METRIC_ENABLE=false SCOPE_EVENT_DEST=file://$influx_path/db/influxd.event $scope_path $influx_path/influxd_stat --config $influx_path/stress_local.conf &
#    $influx_path/influxd_stat --config $influx_path/stress_local.conf &
#    SCOPE_EVENT_HTTP=true SCOPE_EVENT_METRIC=true SCOPE_METRIC_ENABLE=false $scope_path $influx_path/influxd_stat --config $influx_path/stress_local.conf &
else
#    SCOPE_LOG_LEVEL=debug SCOPE_EVENT_HTTP=true SCOPE_EVENT_METRIC=true SCOPE_METRIC_ENABLE=false SCOPE_EVENT_DEST=file://$influx_path/db/influxd.event $scope_path $influx_path/influxd_dyn --config $influx_path/stress_local.conf 2> /dev/null &
    SCOPE_LOG_LEVEL=debug SCOPE_EVENT_HTTP=true SCOPE_EVENT_METRIC=true SCOPE_METRIC_ENABLE=false SCOPE_EVENT_DEST=file://$influx_path/db/influxd.event $scope_path $influx_path/influxd_stat --config $influx_path/stress_local.conf 2> /dev/null &
#	$influx_path/influxd_stat --config $influx_path/stress_local.conf 2>/dev/null &
#    SCOPE_EVENT_HTTP=true SCOPE_EVENT_METRIC=true SCOPE_METRIC_ENABLE=false $scope_path $influx_path/influxd_stat --config $influx_path/stress_local.conf 2> /dev/null&    
fi
    
until test -e "$dbfile" ; do
	sleep 1 
done

sleep 5

i=1

while [ 1 -eq 1 ] ;
do
    echo "   *************** Go stress $i times ***************"

    $influx_path/stress_test insert -r 30s -f
#    $influx_path/stress_test insert -r 30s    
#    SCOPE_EVENT_HTTP=true SCOPE_EVENT_DEST=file://$influx_path/db/influxc.event $scope_path $influx_path/stress_test insert -r 30s -f

    echo "*************** Stress test complete ***************"
    echo ""
    ((i=i+1))
done

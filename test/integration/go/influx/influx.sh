#! /bin/bash

rm -f /tmp/influx*
#nohup ./myscript 0<&- &> my.admin.log.file &
#SCOPE_EVENT_DEST=file:///tmp/influxd.event nohup scope /go/influxd_stat &
./iserver.sh stat
SCOPE_EVENT_DEST=file:///tmp/influxc.event scope ./influx_stress_stat
pkill -f scope

cnt=`grep -c http.req /tmp/influxd.event`
#echo "$cnt"
test "$cnt" -lt 2000 && echo "ERROR: Server" && exit 1
echo "Success: Server"

cnt=`grep -c http.req /tmp/influxc.event`
#echo "$cnt"
test "$cnt" -lt 2000 && echo "ERROR: Server" && exit 1
echo "Success: Client"

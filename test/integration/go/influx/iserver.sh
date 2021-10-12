#! /bin/bash

exec 0>/dev/null
exec 1>/dev/null
exec 2>/dev/null

SCOPE_EVENT_DEST=file:///tmp/influxd.event scope ./influxd_$1 &

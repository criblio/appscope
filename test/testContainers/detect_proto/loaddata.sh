#!/bin/sh
awk -F, 'NR > 1{ print "SET", "\"session_"$2"\"", "\""$1"\"" }' /data/session.csv | redis-cli --pipe
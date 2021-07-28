#! /bin/bash

declare -i ERR=0

wait_for_port() {
    TIMEOUT=${2:-60}
    while ! netstat -an | grep -w ${1} >/dev/null 2>&1; do
        sleep 1
        ((TIMEOUT=TIMEOUT-1))
        if [ $TIMEOUT -le 0 ]; then
            echo >&2 "warn: timed out waiting for port ${1} listener"
            netstat -an | grep -w LISTEN >&2
            return
        fi
    done
    echo 1
}

echo
echo "==============================================="
echo "             Testing Redis                     "
echo "==============================================="
if [ "$(wait_for_port 6379)" ]; then
	ldscope redis-cli SET detect hello
	grep remote_protocol /opt/test-runner/logs/events.log > /dev/null
	ERR+=$?
	grep '"protocol":"Redis"' /opt/test-runner/logs/events.log > /dev/null
	ERR+=$?
	if [ $ERR -eq "0" ]; then
		echo "*************** Redis Success ***************"
	else
		echo "*************** Redis Test Failed ***************"
		cat /opt/test-runner/logs/events.log
	fi
	rm /opt/test-runner/logs/events.log
else
	ERR+=1
	echo "*************** Redis Test Failed ***************"
fi

if [ "x86_64" = "$(uname -m)" ]; then # x86_64 only

	MONGO_ERR=0
	echo
	echo "==============================================="
	echo "             Testing Mongo                     "
	echo "==============================================="
	if [ "$(wait_for_port 27017)" ]; then
		ldscope mongo /opt/test-runner/bin/mongo.js
		grep remote_protocol /opt/test-runner/logs/events.log > /dev/null
		MONGO_ERR+=$?
		grep '"protocol":"Mongo"' /opt/test-runner/logs/events.log > /dev/null
		MONGO_ERR+=$?
		if [ $MONGO_ERR -eq "0" ]; then
			echo "*************** Mongo Success ***************"
		else
			echo "*************** Mongo Test Failed ***************"
			cat /opt/test-runner/logs/events.log
		fi
		rm /opt/test-runner/logs/events.log
		ERR+=$MONGO_ERR
	else
		ERR+=1
		echo "*************** Mongo Test Failed ***************"
	fi

fi # x86_64 only

exit ${ERR}

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
    # Looking for "source":"net.app" events from scoped Redis client.
    # Note that the `protocol[*].detect` entry in scope.yml is `true` to start.

    # Should not get the event with SCOPE_EVENT_NET=false
    rm -f /opt/test-runner/logs/events.log
	SCOPE_EVENT_NET=false ldscope redis-cli SET detect hello >/dev/null 2>&1
	if grep net.app /opt/test-runner/logs/events.log > /dev/null; then
        echo "fail: got event with detect:true,  SCOPE_EVENT_NET=false"
        ERR+=1
    else
        echo "pass: no  event with detect:true,  SCOPE_EVENT_NET=false"
    fi

    # Should get the event when SCOPE_EVENT_NET=true
    rm -f /opt/test-runner/logs/events.log
	SCOPE_EVENT_NET=true ldscope redis-cli SET detect hello >/dev/null 2>&1
	if grep net.app /opt/test-runner/logs/events.log > /dev/null; then
        echo "pass: got event with detect:true,  SCOPE_EVENT_NET=true"
    else
        echo "fail: no  event with detect:true,  SCOPE_EVENT_NET=true"
        ERR+=1
    fi

    # Set detect:false in scope.yml
    sed -i 's/detect: true/detect: false/' /opt/test-runner/bin/scope.yml

    # Should not get the event when SCOPE_EVENT_NET=false
    rm -f /opt/test-runner/logs/events.log
	SCOPE_EVENT_NET=true ldscope redis-cli SET detect hello >/dev/null 2>&1
	if grep net.app /opt/test-runner/logs/events.log > /dev/null; then
        echo "fail: got event with detect:false, SCOPE_EVENT_NET=false"
        ERR+=1
    else
        echo "pass: no  event with detect:false, SCOPE_EVENT_NET=false"
    fi

    # Should not get the event when SCOPE_EVENT_NET=true
    rm -f /opt/test-runner/logs/events.log
	SCOPE_EVENT_NET=true ldscope redis-cli SET detect hello >/dev/null 2>&1
	if grep net.app /opt/test-runner/logs/events.log > /dev/null; then
        echo "fail: got event with detect:false, SCOPE_EVENT_NET=true"
        ERR+=1
    else
        echo "pass: no  event with detect:false, SCOPE_EVENT_NET=true"
    fi

	if [ $ERR -eq 0 ]; then
		echo "*************** Redis Success ***************"
	else
		echo "*************** Redis Test Failed ***************"
		#cat /opt/test-runner/logs/events.log
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
		grep net.app /opt/test-runner/logs/events.log > /dev/null
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

#! /bin/bash

#
# Tests:
# Start a prom exporter, using default ports, run in the background
# Scope an app;  metrics in prom format and metric dest is the prom exporter
# Check for metrics file in tmp dir, validate that size is increasing
# Scrape the exporter, save in a file
# Verify that metric file sizes have decreased, truncation is working.
# Validate metrics from the scrape; match # & { for valid lines
# Kill the scoped app then scrape
# Verify that the metric file has been deleted
#
# Start a test app
# Verify that metrics are being collected; file size
# Kill the prom exporter
# kill the second test app
# Start a prom exporter
# Verify that previous metric files are deleted
#
# Start a prom exporter using alternate port numbers
# Scope an app and verify metrics are being received using the alternate port
# Scrape using the alternate port and verify metrics are received
# Delete the cache file by scraping
#


DEBUG=0 # set this to 1 to capture the EVT_FILE for each test
FAILED_TEST_LIST=""
FAILED_TEST_COUNT=0
LOG_FILE="/tmp/scope.log"
SCRAPE_FILE="/tmp/scrape.log"
CACHE_BASE="scope-metrics-"
ERR=0
CURR_SIZE=0

starttest(){
    CURRENT_TEST=$1
    echo "=============================================="
    echo "             Testing $CURRENT_TEST            "
    echo "=============================================="
    ERR=0

    touch $LOG_FILE
}

endtest(){
    if [ $ERR -eq "0" ]; then
        RESULT=PASSED
    else
        cat $LOG_FILE
        RESULT=FAILED
        FAILED_TEST_LIST+=$CURRENT_TEST
        FAILED_TEST_LIST+=" "
        FAILED_TEST_COUNT=$(($FAILED_TEST_COUNT + 1))
    fi

    echo "******************* $RESULT *******************"
    echo ""
    echo ""
    
    # copy the EVT_FILE to help with debugging
    if (( $DEBUG )) || [ $RESULT == "FAILED" ]; then
        cp $LOG_FILE $LOG_FILE.$CURRENT_TEST
    fi

    rm -f $LOG_FILE
}

# Does a cache file exist
get_cache_file(){
    file_array=`dir /tmp/ | grep $CACHE_BASE`
    found=0

    for entry in $file_array
    do
        if [[ "$entry" = *"$CACHE_BASE"* ]]; then
            found=1
            break
        fi
    done

    echo $found
}

# Return the size of the first cache file found
get_cache_size(){
    file_array=`dir /tmp/ | grep $CACHE_BASE`

    for entry in $file_array
    do
        if [[ "$entry" = *"$CACHE_BASE"* ]]; then
            break
        fi
    done

    filesize=$(stat -c%s "/tmp/$entry")
    echo $filesize
}

validate_cache_increase(){
    local fsize1=$(get_cache_size)

    # It can take up to 10 secs to emit more metrics
    echo "Sleeping in order to ensure the periodic thread runs and we get metrics...."
    sleep 11

    local fsize2=$(get_cache_size)

    if [[ $fsize2 > $fsize1 ]]; then
        echo "validate cache increase; all good"
    else
        echo "ERROR: the cache did not grow as expected"
        ERR+=1
    fi

    CUR_SIZE=$fsize2
}

validate_cache_truncate(){
    fsize=$(get_cache_size)

    if [[ $fsize < $1 ]]; then
        echo "validate cache truncate; all good"
    else
        echo "ERROR: the cache was not truncated as expected: $fsize $1"
        ERR+=1
    fi
}

################# TESTS ################# 

starttest "Prom_Metrics"

# Start a prom exporter
scope prom &

# Create a test app
scope run --metricformat prometheus -m tcp://localhost:9109 -- sleep infinity &
sleep 2

# Are metrics being added to the cache file
validate_cache_increase

# Scrape the metrics
curl -s -o $SCRAPE_FILE http://localhost:9090/metrics

# Did we truncate the cache
validate_cache_truncate "$CUR_SIZE"

# Simple form of data validation
c1=`grep -c \# $SCRAPE_FILE`
c2=`grep -c \{ $SCRAPE_FILE`
if [ $c1 != $c2 ]; then
    echo "ERROR: Validation of data format failed"
    ERR+=1
else
    echo "validate prom format; all good"
fi

# Kill the app, then scrape and see if the cache file is removed
pkill -f sleep
sleep 1
curl -s -o $SCRAPE_FILE http://localhost:9090/metrics
exist=$(get_cache_file)

if [[ $exist == 0 ]]; then
    echo "validate cache removed on process exit; all good "
else
    echo "ERROR: Cache removed on process exit"
    ERR+=1
fi

rm $SCRAPE_FILE
pkill -f scope
pkill -f sleep

endtest

###############################################
starttest "Remove_Initial_Cache"

# Start a prom exporter
scope prom &

# Create a test app
scope run --metricformat prometheus -m tcp://localhost:9109 -- sleep infinity &
sleep 2

# Kill the app and the exporter
pkill -f scope
pkill -f sleep

# Start a prom exporter and verify that the previous cache has been removed
scope prom &
sleep 2

exist=$(get_cache_file)

if [[ $exist == 0 ]]; then
    echo "validate cache removed on prom start; all good "
else
    echo "ERROR: Cache removed on prom start"
    ERR+=1
fi

pkill -f scope
pkill -f sleep

endtest

###############################################
starttest "Alternate_Ports"

# Start a prom exporter with specific ports
scope prom --sport 9099 --mport 9110 &
ERR+=$?

# Create a test app
scope run --metricformat prometheus -m tcp://localhost:9110 -- sleep infinity &
sleep 2

# Are metrics being added to the cache file
validate_cache_increase

# Scrape the metrics
curl -s -o $SCRAPE_FILE http://localhost:9099/metrics
ERR+=$?

# Simple form of data validation
c1=`grep -c \# $SCRAPE_FILE`
c2=`grep -c \{ $SCRAPE_FILE`
if [ $c1 != $c2 ]; then
    echo "ERROR: Validation of data format with alternate ports failed"
    ERR+=1
else
    echo "validate prom format with alternate ports; all good"
fi

rm $SCRAPE_FILE
pkill -f sleep
sleep 1

# Delete the cache file
curl -s http://localhost:9099/metrics
ERR+=$?
sleep 2

pkill -f scope

endtest

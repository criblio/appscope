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
# Verify that metrics are being collected; file size
# Kill the prom exporter
# kill the second test app
# start a prom exporter using alternate port numbers
# Verify that previous metric file are deleted
# Scope an app and verfiy mtrics are being received using the alternate port
# Scrape using the alternate port and verify metrics are received
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

# Return the size of the first cache file found
get_cache_size(){
    file_array=`dir /tmp/ | grep $CACHE_BASE`

    for entry in $file_array
    do
        #echo "$entry"
        #echo "checking $entry and $CACHE_BASE"
        if [[ "$entry" = *"$CACHE_BASE"* ]]; then
            #echo "Found $entry"
            break
        fi
    done

    filesize=$(stat -c%s "/tmp/$entry")
    echo $filesize
}

validate_cache_increase(){
    local fsize1=$(get_cache_size)
    #echo $fsize1
    echo "Starting V_C_I"
    # it can take up to 10 secs to emit more metrics
    sleep 11

    local fsize2=$(get_cache_size)
    #echo $fsize2
    echo "VCI"

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
    #echo $fsize
    #echo "v_c_t: p1 $1"


    if [[ $fsize < $1 ]]; then
        echo "validate cache truncate; all good"
    else
        echo "ERROR: the cache was not truncated as expected: $fsize $1"
        ERR+=1
    fi
}

################# TESTS ################# 

# Init test
starttest "Prom_Metrics"

# Start a prom exporter
scope prom &

# Create a test app
scope run --metricformat prometheus -m tcp://localhost:9109 -- sleep infinity &
sleep 2

# Are metrics being added to the cache file
echo "V_C_I..."
validate_cache_increase
#cur_size=$(validate_cache_increase)
echo "VCI return: $CUR_SIZE"

# Scrape the metrics
echo "starting: curl -o $SCRAPE_FILE http://localhost:9090/metrics"
curl -o $SCRAPE_FILE http://localhost:9090/metrics

# Did we truncate the cache
validate_cache_truncate "$CUR_SIZE"

# Simple form of data validation
c1=`grep -c \# $SCRAPE_FILE`
c2=`grep -c \{ $SCRAPE_FILE`
if [ $c1 != $c2 ]; then
    echo "ERROR: Validation of data format failed"
    ERR+=1
fi

rm $SCRAPE_FILE
pkill -f scope
pkill -f sleep

endtest

#!/bin/bash
DEBUG=0  # set this to 1 to capture the DEST_FILE for each test

FAILED_TEST_LIST=""
FAILED_TEST_COUNT=0
DEST_FILE="/tmp/output_dest_file"
FILE_SOCKET="/tmp/file_socket"
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

    # copy the DEST_FILE to help with debugging
    if (( $DEBUG )) || [ $RESULT == "FAILED" ]; then
        cp -f $DEST_FILE $DEST_FILE.$CURRENT_TEST
    fi

    rm -f $DEST_FILE
}

export SCOPE_PAYLOAD_ENABLE=true
export SCOPE_PAYLOAD_HEADER=true

### change current directory
cd /opt/test-runner

#
# scope metricdest file socket
#

starttest metricdest_file_socket

nc -lU $FILE_SOCKET > $DEST_FILE &
ERR+=$?

scope run --metricdest=unix://$FILE_SOCKET ls
ERR+=$?

count=$(grep '"type":"metric"' $DEST_FILE | wc -l)
if [ $count -eq 0 ] ; then
    ERR+=1
fi

count=$(grep '"type":"evt"' $DEST_FILE | wc -l)
if [ $count -ne 0 ] ; then
    ERR+=1
fi

endtest

#
# scope eventdest file socket
#

starttest eventdest_file_socket

nc -lU $FILE_SOCKET > $DEST_FILE &
ERR+=$?

scope run --eventdest=unix://$FILE_SOCKET ls
ERR+=$?

count=$(grep '"type":"metric"' $DEST_FILE | wc -l)
if [ $count -ne 0 ] ; then
    ERR+=1
fi

count=$(grep '"type":"evt"' $DEST_FILE | wc -l)
if [ $count -eq 0 ] ; then
    ERR+=1
fi

endtest

#
# scope cribldest file socket
#

starttest cribldest_file_socket

nc -lU $FILE_SOCKET > $DEST_FILE &
ERR+=$?

scope run --cribldest=unix://$FILE_SOCKET ls
ERR+=$?

count=$(grep '"type":"metric"' $DEST_FILE | wc -l)
if [ $count -eq 0 ] ; then
    ERR+=1
fi

count=$(grep '"type":"evt"' $DEST_FILE | wc -l)
if [ $count -eq 0 ] ; then
    ERR+=1
fi

endtest

unset SCOPE_PAYLOAD_ENABLE
unset SCOPE_PAYLOAD_HEADER

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

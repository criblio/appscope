#! /bin/bash

DEBUG=0  # set this to 1 to capture the EVT_FILE for each test

FAILED_TEST_LIST=""
FAILED_TEST_COUNT=0

PRELOAD=`env | grep LD_PRELOAD`
EVT_FILE="/opt/test-runner/logs/events.log"

starttest(){
    CURRENT_TEST=$1
    echo "==============================================="
    echo "             Testing $CURRENT_TEST             "
    echo "==============================================="
    export $PRELOAD
    ERR=0
}

evaltest(){
    echo "             Evaluating $CURRENT_TEST"
    unset LD_PRELOAD
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


starttest Tomcat
/opt/tomcat/bin/catalina.sh run &
evaltest

#until [ "`curl --ciphers rsa_aes_128_sha -k --silent --connect-timeout 1 -I https://localhost:8443 | grep 'Coyote'`" != "" ];
until [ "`curl -k --silent --connect-timeout 1 -I https://localhost:8443 | grep 'Coyote'`" != "" ];
do
    echo waiting for tomcat...
    sleep 1
done

sleep 2
grep http-req $EVT_FILE > /dev/null
ERR+=$?

grep http-resp $EVT_FILE > /dev/null
ERR+=$?

grep HTTP $EVT_FILE > /dev/null
ERR+=$?
endtest


starttest SSLSocketClient
cd /opt/javassl
java -Djavax.net.ssl.trustStore=/opt/tomcat/certs/tomcat.p12 -Djavax.net.ssl.trustStorePassword=changeit -Djavax.net.ssl.trustStoreType=pkcs12 SSLSocketClient > /dev/null
evaltest
grep http-req $EVT_FILE > /dev/null
ERR+=$?

grep http-resp $EVT_FILE > /dev/null
ERR+=$?

grep HTTP $EVT_FILE > /dev/null
ERR+=$?
endtest

/opt/tomcat/bin/catalina.sh stop

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


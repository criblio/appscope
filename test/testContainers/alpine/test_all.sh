#!/bin/bash

ERR=0


export SCOPE_EVENT_DEST=file:///go/events.log
/go/test_go.sh
ERR+=$?

export ENV SCOPE_EVENT_DEST=file:///opt/test/logs/events.log
/opt/test/bin/test_tls.sh
ERR+=$?

exit ${ERR}

#!/bin/bash
if [ -n "${DEMOUSER_NAME}" ]; then
    /bin/sed -i 's/const name = undefined;/const name = "'"${DEMOUSER_NAME}"'";/' $SPLUNK_HOME/etc/apps/cribl/appserver/static/dashboard.js
fi
if [ -n "${DEMOUSER_EMAIL}" ]; then
    /bin/sed -i 's/const email = undefined;/const email = "'"${DEMOUSER_EMAIL}"'";/' $SPLUNK_HOME/etc/apps/cribl/appserver/static/dashboard.js
fi

cat $SPLUNK_HOME/etc/apps/cribl/appserver/static/dashboard.js

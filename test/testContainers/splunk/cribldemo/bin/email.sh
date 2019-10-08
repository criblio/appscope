#!/bin/bash

# Borrowed from https://coderwall.com/p/3vqf2g/send-emails-via-amazon-ses-with-bash-and-curl
# and modified

export CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

sleep 60

if [ -f email.lock ] || [ -z "${DEMOUSER_EMAIL}" ]; then
    echo "Email.lock present or DEMOUSER_EMAIL not set"
    exit 0
fi

base64="base64 -w 0"
if [ "$(uname)" = "Darwin" ]; then
    base64="base64 -b 0"
fi

FIRSTNAME=$(echo ${DEMOUSER_NAME} | awk '{print $1}')
DEMOUSER_IP=$(LD_LIBRARY_PATH= curl -s ifconfig.co)

TO="${DEMOUSER_EMAIL}"
FROM="Cribl Sandbox <hello@cribl.io>"
SUBJECT="Your Cribl Sandbox is ready!"
read -d '' MESSAGE <<_EOF_
${FIRSTNAME},

Thanks for trying the Cribl Sandbox! The sandbox contains everything you need to learn how 
Cribl can help you control your data. The sandbox contains a running instance of Splunk with a 
data generator running behind the scenes to put some live demo data into the system. Cribl is running 
to transform the data. Inside Splunk, we've put a demo app which contains examples of a number of use 
cases you can implement with Cribl. It's interactive and live. Please play with, modify it, break it, 
you can always stand up a new one!

For Splunk UI go to: http://${DEMOUSER_IP}:8000/en-US/account/insecurelogin?loginType=splunk&username=admin&password=cribldemo

Once logged in, you'll be dropped right into the Cribl demo app. 

For Cribl UI go to: http://${DEMOUSER_IP}:9000/login?username=admin&password=cribldemo

The sandbox environment will automatically shut down in about 1 hour.

If you're interested in running the bits in your Splunk environment, please reach out to us
at hello@cribl.io or go to https://cribl.io/download.

Thanks!
Cribl Founding Team
_EOF_

date="$(date +"%a, %d %b %Y %H:%M:%S %Z")"
priv_key="${AWS_SECRET_ACCESS_KEY}"
access_key="${AWS_ACCESS_KEY_ID}"
signature="$(echo -n "$date" | $SPLUNK_HOME/bin/splunk cmd openssl dgst -sha256 -hmac "$priv_key" -binary | $base64)"
auth_header="X-Amzn-Authorization: AWS3-HTTPS AWSAccessKeyId=$access_key, Algorithm=HmacSHA256, Signature=$signature"
endpoint="https://email.us-west-2.amazonaws.com/"

action="Action=SendEmail"
source="Source=$FROM"
to="Destination.ToAddresses.member.1=$TO"
subject="Message.Subject.Data=$SUBJECT"
message="Message.Body.Text.Data=$MESSAGE"

LD_LIBRARY_PATH= curl -X POST -H "Date: $date" -H "$auth_header" --data-urlencode "$message" --data-urlencode "$to" --data-urlencode "$source" --data-urlencode "$action" --data-urlencode "$subject"  "$endpoint"
touch email.lock

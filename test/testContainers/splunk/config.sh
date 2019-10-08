#!/bin/bash

if [ -n "$CRIBL_ROUTING_DEMO" ]; then
	cat <<-EOF >> $SPLUNK_HOME/etc/apps/cribl/local/inputs.conf
		[splunktcp://9997]
		connection_host = ip

		[splunktcp://9998]
		connection_host = ip
	EOF
fi

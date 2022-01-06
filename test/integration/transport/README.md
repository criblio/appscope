# Integration Tests for Back-End Transports

We're using Cribl LogStream setup with a few sourceswe can connect to.

* TCP:10090 is configured as an AppScope source using TLS and the cert/key pair in /tmp
* TCP:10091 is another AppScope source using plain TCP
* TCP:10070 is a TCP JSON source
* UDP:8125 is a UDP metrics source

We have a Filesystem destination set to produce files under `/tmp/out` with a subdirectory for each source; `appscope:in_appscope_tls`, `appscope:in_appscope_tcp`, `tcp_json:in_tcp_json`, and `metrics:in_metrics`. The destination is set to close files and move them to where they go after they've been idle for 5 seconds so tests need to wait 5+ seconds before looking for the files they expect to get.

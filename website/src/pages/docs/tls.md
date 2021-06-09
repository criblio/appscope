---
title: TLS
---

## Using TLS for Secure Connections
---

AppScope supports TLS over TCP connections. LogStream, events, and metrics can all use TLS when connecting to AppScope over TCP. And AppScope can use TLS when connecting to LogStream or another application.

To see the TLS-related environment variables, run the command: `ldscope --help configuration | grep TLS`

In the `scope.yml` config file, the `transport` definition includes an optional `tls` element. See [Config Files](/docs/config-files).

### AppScope in Cribl.Cloud Uses TLS by Default

In Cribl.Cloud, AppScope communicates with LogStream using a dedicated TLS over TCP connection *by default*, on port 10090. The cloud infrastructure automatically enables TLS. You do not need to configure TLS for the AppScope Source in LogStream.

For communication in cleartext (not using TLS), use port 10091. 

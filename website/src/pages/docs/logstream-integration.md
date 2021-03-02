---
title: Integrating with LogStream
---

## Integrating with LogStream
---

AppScope will connect to LogStream. A single configuration element is used to define a LogStream connection. The environment variable SCOPE_LOGSTREAM defines a LogStream connection:
SCOPE_LOGSTREAM=hostname:port
SCOPE_LOGSTREAM=ip:port
SCOPE_LOGSTREAM=tcp://ip:port

Default port is 10090

A host name or IPv4 address is supplied along with a port number. If no port number is supplied, the default port 10090 will be used. Optionally, the syntax “tcp://ip:port” is supported to define an IP address.

When a LogStream connection is established, several configuration parameters are set and will override settings in a configuration file or environment variable. Any override of configuration is logged to the default AppScope log file.  The configuration settings that are not variable and potentially overridden include: 
Metrics are enabled
Metrics format is set to ndjson
Events are enabled
Transport for events, metrics and payloads utilize the LogStream connection

Configuration elements that are enabled by default when a LogStream connection is defined and can be overridden by a configuration file or environment variables:
Event watch types are enabled
File
Console
FS
Net
HTTP
DNS

All other configuration elements remain default.

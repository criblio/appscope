---
title: Integrating with LogStream
---

## Integrating with LogStream

AppScope can easily connect to Cribl LogStream ([overview](https://cribl.io/product/) | [download](https://cribl.io/download/) | [docs](https://docs.cribl.io/docs/welcome)).

To define a LogStream connection, the only configuration element needed is setting the `SCOPE_CRIBL` environment variable. For example:

```
SCOPE_CRIBL=tcp://ip:port
```

### Requirements

You must supply a host name or IPv4 address, along with an optional port number. If you omit the port number, AppScope will attempt to connect on the default port `10090`. 

### Parameter Overrides

When AppScope establishes a LogStream connection, this sets several AppScope configuration parameters, some of which override settings in any configuration file or environment variable. Any configuration override is logged to AppScope's default log file. 

Configuration settings that are potentially overridden include: 

- Metrics are enabled.
- Metrics format is set to `ndjson`.
- Events are enabled.
- Transport for events, metrics, and payloads use the LogStream connection.

The following configuration elements are enabled by default when a LogStream connection is defined, but these can be overridden by a configuration file or by environment variables:

- Event watch types
- File
- Console
- FS
- Net
- HTTP
- DNS

Other configuration elements are not modified by a LogStream connection.

## Using TLS for Secure Connections

AppScope supports TLS over TCP connections.

See [Using TLS for Secure Connections](/docs/tls).

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

On the LogStream side, a built-in [AppScope Source](https://docs.cribl.io/docs/sources-appscope) receives data from AppScope by default. You can change its configuration or create additional AppScope Sources as needed.

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
- HTTP/1.1 and HTTP/2.0
- DNS

Other configuration elements are not modified by a LogStream connection.

### HTTP Traffic

AppScope streams HTTP payloads to LogStream, whether the payloads originate as HTTP/1.1 or HTTP/2 traffic. LogStream then converts the payloads to HTTP events.

Separately, the AppScope can convert HTTP/1.1 (not HTTP/2) payloads to HTTP events. This is mainly useful for streaming HTTP events to destinations **other than** LogStream.

## Using TLS for Secure Connections

AppScope supports TLS over TCP connections.

See [Using TLS for Secure Connections](/docs/tls).

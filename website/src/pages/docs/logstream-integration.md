---
title: Integrating with LogStream
---

## Integrating with LogStream

AppScope can easily connect to Cribl LogStream ([overview](https://cribl.io/product/) | [cloud](https://cribl.cloud/) | [download](https://cribl.io/download/) | [docs](https://docs.cribl.io/docs/welcome)).

To define a LogStream connection, simply set the `SCOPE_CRIBL` environment variable specifying a transport type, IP address, and port. For example:

```
SCOPE_CRIBL=tcp://127.0.0.1:9109
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

The following configuration elements are event watch types which are enabled by default when a LogStream connection is defined:

- `console`
- `dns`      
- `file`
- `fs`
- `http`
- `net`

These event watch types can be overridden by a configuration file or by environment variables.

Other configuration elements are not modified by a LogStream connection.

### HTTP Traffic

The AppScope Source streams HTTP payloads to LogStream, whether the payloads originate as HTTP/1.1 or HTTP/2 traffic. LogStream then converts the payloads to HTTP events.

Separately, the AppScope library can convert HTTP/1.1 (not HTTP/2) payloads to HTTP events. This is mainly useful for streaming HTTP events to destinations **other than** LogStream.

## Using TLS for Secure Connections

AppScope supports TLS over TCP connections.

See [Using TLS for Secure Connections](/docs/tls).

---
title: Integrating with LogStream
---

## Integrating with LogStream

AppScope can easily connect to Cribl LogStream ([overview](https://cribl.io/product/) | [cloud](https://cribl.cloud/) | [download](https://cribl.io/download/) | [docs](https://docs.cribl.io/docs/welcome)).

To define a TLS-encrypted connection to LogStream Cloud, just set the `SCOPE_CRIBL_CLOUD` environment variable, specifying a transport type, a host name or IPv4 address, and a port number. 

For example:

```
SCOPE_CRIBL_CLOUD=tcp://in.logstream.<cloud_instance>.cribl.cloud:10090
```
By default, LogStream Cloud instances have port `10090` configured to use [TLS](/docs/tls) over TCP, and a built-in AppScope Source to receive data from AppScope. You can change the [AppScope Source configuration](https://docs.cribl.io/docs/sources-appscope), or create additional AppScope Sources, as needed.

This is the easiest way to integrate AppScope with LogStream. It is also possible to connect to LogStream Cloud [unencrypted](#cloud-unencrypted); or, to LogStream [on-prem](#on-prem), encrypted or not.

### Parameter Overrides

When AppScope establishes a LogStream connection, this sets several AppScope configuration parameters, some of which override settings in any configuration file or environment variable.

These overrides include: 

- Metrics format is set to `ndjson`.
- Transport for events, metrics, and payloads use the LogStream connection.
- Log level, is set to `warning` if originally configured as `error` or `none`.

Any configuration override is logged to AppScope's default log file. 

If you have Events or Metrics enabled (or disabled), they stay how you have them. (Before 1.0.0 they would be set to enabled, as an override). You still have control via environment variables or the config file.

<span id="cloud-unencrypted"> </span>

### Connecting to LogStream Cloud Unencrypted 

To define an **unencrypted** connection to a LogStream Cloud instance, set the `SCOPE_CRIBL` environment variable (**not** `SCOPE_CRIBL_CLOUD`) and specify port `10091`.

For example:

```
SCOPE_CRIBL=tcp://in.logstream.<cloud_instance>.cribl.cloud:10091
```

<span id="on-prem"> </span>

### Connecting to LogStream On-Prem 

An on-prem instance of LogStream has an AppScope Source built in. However, by default: 

- The Source itself is disabled. 
- The Source listens on port 10090.
- On the Source, TLS is disabled by default.

To connect from AppScope, you'll need to [configure](https://docs.cribl.io/logstream/sources-appscope) the AppScope Source in LogStream. This includes
enabling the Source; configuring it to listen on the desired port; and, enabling and configuring TLS if desired.

Then, to define a connection to the on-prem LogStream instance, set `SCOPE_CRIBL`.  

For example:

```
SCOPE_CRIBL=tcp://127.0.0.1:10090
```

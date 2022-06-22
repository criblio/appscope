---
title: Integrating with Cribl
---

# Integrating with Cribl

By default, AppScope is configured to connect to an instance of Cribl Edge ([docs](https://docs.cribl.io/edge/)) running on the same host. Cribl Edge, in turn, connects to Cribl Stream over the network. This arrangement can support scaling to large numbers of scoped processes.

AppScope can also easily connect to Cribl Stream ([overview](https://cribl.io/product/) | [cloud](https://cribl.cloud/) | [download](https://cribl.io/download/) | [docs](https://docs.cribl.io/docs/welcome)) directly.

## Integrating with Cribl Edge

AppScope will connect with Cribl Edge when the `cribl` backend is enabled with the `edge` transport type. These are default settings for the AppScope library, which is [best suited](/docs/working-with) for the kinds of longer-running, planned monitoring procedures that tend to scale, and where the AppScope-Cribl Edge combination excels. 

The CLI, by contrast, writes to the local filesystem by default; and the CLI is best suited for ad hoc exploration that's less likely to scale to large numbers of scoped processes.

To define a connection to Cribl Edge with the AppScope library, just set the `SCOPE_CRIBL` environment variable to `edge`. 

For example:

```
LD_PRELOAD=./libscope.so SCOPE_CRIBL=edge ls -al
```

### Scaling to Many Scoped Processes {#scaling-scoped-processes}

The number of scoped processes can become very high in two kinds of situations:
1. Where the application that one instance of AppScope is monitoring spawns a large number of processes - for example, scoping a build process; and,
2. Where many instances of AppScope are monitoring many instances of an application in a distributed architecture - for example, monitoring Apache in a server farm.

In either case, it can help when AppScope connects to Cribl Edge on the same host, and Cribl Edge then connects to Cribl Stream over the network.

To understand why this is, consider these facts: 
* Within the same host, AppScope connects to Cribl Edge over unix sockets. 
* The number of sockets is only limited by number of processes that the host can support. 
* Then, Cribl Edge only requires a **single network connection** to send all of the data on to Cribl Stream.

This results in dramatically lower numbers of network connections needed than if AppScope connected to Cribl Stream over the network without Cribl Edge as a proxy. Each network connection requires a port, and a large number of connections could exhaust the supply of ports available on the Cribl Stream host.

Sending data through Cribl Edge has other benefits: Cribl Edge is capable of filtering, aggregating, and reducing data in the same manner as Cribl Stream, and it can be efficient to perform these operations on the data before sending it over the wire. 

## Integrating with Cribl Stream

To define a TLS-encrypted connection to Cribl Stream Cloud, just set the `SCOPE_CRIBL_CLOUD` environment variable, specifying a transport type, a host name or IPv4 address, and a port number. 

For example:

```
SCOPE_CRIBL_CLOUD=tcp://in.logstream.<cloud_instance>.cribl.cloud:10090
```
By default, Cribl Stream Cloud instances have port `10090` configured to use [TLS](/docs/tls) over TCP, and a built-in AppScope Source to receive data from AppScope. You can change the [AppScope Source configuration](https://docs.cribl.io/docs/sources-appscope), or create additional AppScope Sources, as needed.

This is the easiest way to integrate AppScope with Cribl Stream. It is also possible to connect to Cribl Stream Cloud [unencrypted](#cloud-unencrypted); or, to Cribl Stream [on-prem](#on-prem), encrypted or not.

### Parameter Overrides

When AppScope establishes a Cribl Stream connection, this sets several AppScope configuration parameters, some of which override settings in any configuration file or environment variable.

These overrides include: 

- Metrics format is set to `ndjson`.
- Transport for events, metrics, and payloads use the Cribl Stream connection.
- Log level, is set to `warning` if originally configured as `error` or `none`.

Any configuration override is logged to AppScope's default log file.

If you have events or metrics enabled (or disabled), they retain this setting. (Before v.1.0.0, both would be set to enabled, as an override). You still have control via environment variables or the config file.

<span id="cloud-unencrypted"> </span>

### Connecting to Cribl Stream Cloud Unencrypted 

To define an **unencrypted** connection to a Cribl Stream Cloud instance, set the `SCOPE_CRIBL` environment variable (**not** `SCOPE_CRIBL_CLOUD`) and specify port `10091`.

For example:

```
SCOPE_CRIBL=tcp://in.logstream.<cloud_instance>.cribl.cloud:10091
```

<span id="on-prem"> </span>

### Connecting to Cribl Stream On-Prem 

An on-prem instance of Cribl Stream has an AppScope Source built in. However, by default: 

- The Source itself is disabled. 
- The Source listens on port 10090.
- On the Source, TLS is disabled by default.

To connect from AppScope, you'll need to [configure](https://docs.cribl.io/logstream/sources-appscope) the AppScope Source in Cribl Stream. This includes
enabling the Source; configuring it to listen on the desired port; and, enabling and configuring TLS if desired.

Then, to define a connection to the on-prem Cribl Stream instance, set `SCOPE_CRIBL`.  

For example:

```
SCOPE_CRIBL=tcp://127.0.0.1:10090
```

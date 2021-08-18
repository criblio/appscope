---
title: Using TLS for Secure Connections
---

## Using TLS for Secure Connections

AppScope supports TLS over TCP connections: 

- AppScope can use TLS when connecting to LogStream or another application (including its events and metrics destinations).
- LogStream can use TLS when connecting to AppScope over TCP.

To see the TLS-related environment variables, run the command: `ldscope --help configuration | grep TLS`

In the `scope.yml` config file, the `transport` definition includes an optional `tls` element. See [Config Files](/docs/config-files).

## Using TLS in Cribl.Cloud

In Cribl.Cloud, when communicating with LogStream, AppScope uses TLS by default.

Within Cribl.Cloud, a front-end load balancer (reverse proxy) handles the encrypted TLS traffic and relays it to the AppScope Source port in LogStream. The connection from the load balancer to LogStream does **not** use TLS, and you should not enable TLS on the [AppScope Source](https://docs.cribl.io/docs/sources-appscope) in LogStream. No changes in LogStream configuration are needed.

AppScope connects to port 10090 of the Cribl.Cloud Ingest Endpoint. Use the tenant hostname you were assigned when you joined Cribl.Cloud.

### CLI usage

Use scope with the `-c` option:

```
scope -c tls://host:10090
```

### Configuration for `LD_PRELOAD` or `ldscope`

To connect AppScope to a LogStream Cloud instance using TLS: 

1. Enable the `tls` element in `scope.yml`.
1. Connect to port 10090 on your Cribl.Cloud Ingest Endpoint.

To enable TLS in `scope.yml`, adapt the example below to your environment:

```
cribl:
  enable: true
  transport:
    type: tcp  # don't use tls here, use tcp and enable tls below
    host: in.logstream.example-tenant.cribl.cloud
    port: 10090 # cribl.cloud's port for the TLS AppScope Source
    tls:
      enable: true
      validateserver: true
      cacertpath: ''
```

## Adapting to Network Latency

When you connect AppScope to LogStream or another application, you can usually assume that the process you are scoping will last long enough for AppScope to make the network connection and stream the data to the destination.

This might not hold true if the command or application being scoped exits quickly and network connection latency is relatively high. Under these conditions, events and metrics can be dropped and never reach the destination.

To mitigate this problem, you can set the `SCOPE_CONNECT_TIMEOUT_SECS` environment variable (which is unset by default) to `1`. Then, the AppScope library will keep the scoped process open for 1 second, which should be enough time for AppScope to connect and stream data over the network.

## Scoping Without TLS

If you prefer to communicate without encryption, connect to port 10091 instead of port 10090.

If it is enabled, disable the `tls` element in `scope.yml`.

If connecting to LogStream in Cribl.Cloud, no changes in LogStream configuration are needed.

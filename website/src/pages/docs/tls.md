---
title: Using TLS for Secure Connections
---

## Using TLS for Secure Connections

AppScope supports TLS over TCP connections: 

- AppScope can use TLS when connecting to LogStream or another application (including its events and metrics destinations).
- LogStream can use TLS when connecting to AppScope over TCP.

In the `scope.yml` config file, set the `transport : tls` element to `true` to enable TLS. See [Config Files](/docs/config-file).

To see the TLS-related environment variables, run the command: 

```
ldscope --help configuration | grep TLS
```

## Using TLS in Cribl.Cloud

AppScope uses TLS by default to communicate with LogStream in Cribl.Cloud. LogStream has an AppScope Source ready to use out-of-the-box.

Within Cribl.Cloud, a front-end load balancer (reverse proxy) handles the encrypted TLS traffic and relays it to the AppScope Source port in LogStream. The connection from the load balancer to LogStream does **not** use TLS, and you should not enable TLS on the [AppScope Source](https://docs.cribl.io/docs/sources-appscope) in LogStream. No changes in LogStream configuration are needed.

AppScope connects to your Cribl.Cloud Ingest Endpoint on port 10090. The Ingest Endpoint URL is always the same except for the Cribl.Cloud Organization ID, which LogStream uses in the hostname portion, in the following way:

```
https://in.logstream.<organization-ID>.cribl.cloud:10090
```

If you **disable** TLS, the port is 10091.

### CLI usage

Use scope with the `-c` option:

```
scope -c tls://host:10090
```

### Configuration for `LD_PRELOAD` or `ldscope`

To connect AppScope to a LogStream Cloud instance using TLS: 

1. Enable the `transport : tls` element in `scope.yml`.
1. Connect to port 10090 on your Cribl.Cloud Ingest Endpoint.

To enable TLS in `scope.yml`, adapt the example below to your environment:

```
cribl:
  enable: true
  transport:
    type: tcp  # don't use tls here, use tcp and enable tls below
    host: in.logstream.example-organization.cribl.cloud
    port: 10090 # cribl.cloud's port for the TLS AppScope Source
    tls:
      enable: true
      validateserver: true
      cacertpath: ''
```

## Scoping Without TLS

If you prefer to communicate without encryption, connect to port 10091 instead of port 10090.

If it is enabled, disable the `tls` element in `scope.yml`.

If connecting to LogStream in Cribl.Cloud, no changes in LogStream configuration are needed.

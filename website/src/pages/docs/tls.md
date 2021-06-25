---
title: TLS
---

## Using TLS for Secure Connections

AppScope supports TLS over TCP connections: 

- LogStream can use TLS when connecting to AppScope over TCP.
- AppScope can use TLS when connecting to LogStream or another application.

To see the TLS-related environment variables, run the command: `ldscope --help configuration | grep TLS`

In the `scope.yml` config file, the `transport` definition includes an optional `tls` element. See [Config Files](/docs/config-files).

## Using TLS in Cribl.Cloud

In Cribl.Cloud, AppScope can communicate with LogStream either using TLS (the default), or in cleartext.

Within Cribl.Cloud, a front-end load balancer (reverse proxy) handles the encrypted TLS traffic and relays it to the AppScope Source port in LogStream. The connection from the load balancer to LogStream does *not* use TLS, and you should not enable TLS on the AppScope Source in LogStream.

In Cribl.Cloud the Ingest Endpoint uses port 10090 for TLS and port 10091 for cleartext.

### CLI usage

Use scope with the `-c` option:

```
scope -c tls://host:10090
```

### `LD_PRELOAD` or `ldscope`

To connect AppScope to a LogStream Cloud instance using TLS: 

1. Enable the `tls` element in `scope.yml` 
1. Connect to port 10090 on your Cribl.Cloud Ingest Endpoint

To enable TLS in `scope.yml`, adapt the example below to your environment:

```
cribl:
  enable: true
  transport:
    type: tcp  # don't use tls here, use tcp and enable tls below
    host: in.logstream.example-tenant.cribl.cloud
    port: 10090 # cribl.cloud's port for the TLS AppSCope Source
    tls:
      enable: true
      validateserver: true
      cacertpath: ''
```

To connect AppScope to a LogStream Cloud instance *without* TLS: 

1. Disable the `tls` element in `scope.yml` 
1. Connect to port 10091 on your Cribl.Cloud Ingest Endpoint


---
title: Using TLS for Secure Connections
---

## Using TLS for Secure Connections

AppScope supports TLS over TCP connections. Here's how that works:

- AppScope can use TLS when connecting to Cribl Stream or another application (including its events and metrics destinations). 
- Once AppScope establishes the connection, data can flow over that connection **in both directions**. 
- This means that when you tell AppScope to connect using TLS, the connection is secured by TLS in both directions.

For security's sake, AppScope never opens ports, nor does it listen for or allow incoming connections.

To enable TLS: In the `scope.yml` [config file](/docs/config-file), set the `transport : tls : enable` element to `true`.

To see the TLS-related environment variables, run the command: 

```
ldscope --help configuration | grep TLS
```

## Using TLS in Cribl.Cloud

AppScope uses TLS by default to communicate with Cribl Stream on Cribl.Cloud. Cribl Stream has an AppScope Source ready to use out-of-the-box.

Within Cribl.Cloud, a front-end load balancer (reverse proxy) handles the encrypted TLS traffic and relays it to the AppScope Source port in Cribl Stream. The connection from the load balancer to Cribl Stream does **not** use TLS, and you should not enable TLS on the [AppScope Source](https://docs.cribl.io/docs/sources-appscope) in Cribl Stream. No changes in Cribl Stream configuration are needed.

AppScope connects to your Cribl.Cloud Ingest Endpoint on port 10090. The Ingest Endpoint URL is always the same except for the Cribl.Cloud Organization ID, which Cribl Stream uses in the hostname portion, in the following way:

```
https://in.logstream.<organization-ID>.cribl.cloud:10090
```

If you **disable** TLS, the port is 10091.

### CLI usage

Use `scope run` with the `-c` option:

```
scope run -c tls://host:10090
```

### Configuration for `LD_PRELOAD` or `ldscope`

To connect AppScope to a Cribl.Cloud-managed instance of Cribl Stream using TLS: 

1. Enable the `transport : tls : enable` element in `scope.yml`.
1. Connect to port 10090 on your Cribl.Cloud Ingest Endpoint.

To enable TLS in `scope.yml`, adapt the example below to your environment:

```
cribl:
  enable: true
  transport:
    type: tcp  # don't use tls here, use tcp and enable tls below
    host: in.Cribl Stream.example-organization.cribl.cloud
    port: 10090 # cribl.cloud's port for the TLS AppScope Source
    tls:
      enable: true
      validateserver: true
      cacertpath: ''
```

## Scoping Without TLS

If you prefer to connect to Cribl Stream on Cribl.Cloud without encryption, connect to port 10091 instead of port 10090, and disable the `tls` element in `scope.yml`.

No changes in Cribl Stream configuration are needed.

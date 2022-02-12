---
title: Data Routing
---

## Data Routing

AppScope gives you multiple ways to route data. The basic data routing operations are:

- Routing both events and metrics to LogStream Cloud
- Routing events to a destination
- Routing metrics to a destination

For each of these operations, the CLI has a command-line option, the config file has a setting, and the AppScope library has an environment variable. In a few cases, two options, settings, or environment variables are required.

You can route data to files, local unix sockets, or network destinations. 

<!-- resolve contradiction: network or cloud? 

Set Cribl LogStream destination for metrics & events (host:port defaults to tls://)

-->

### Routing Data to LogStream Cloud

How to route events and metrics (together, not individually) to a cloud destination:

#### CLI Routing

Use the `-c` or `--cribldest` option. You need this option because, by default, CLI routes data to the local filesystem. For example:

```
scope run -c tcp://127.0.0.1:10091 -- curl https://wttr.in/94105
```

This usage works with `scope run`, `scope attach`, `scope watch`, `scope k8s`, and `scope service`. It does **not** work with `scope metrics`, where the `-c` option stands for `--cols`, telling the CLI to output data in columns instead of rows.

#### AppScope Library Routing

Use the `SCOPE_CRIBL_CLOUD` environment variable to define a TLS-encrypted connection to LogStream Cloud. You specify a transport type, a host name or IPv4 address, and a port number, for example:

```
SCOPE_CRIBL_CLOUD=tcp://in.logstream.<cloud_instance>.cribl.cloud:10090
```

As a convenience, when you set `SCOPE_CRIBL_CLOUD`, AppScope automatically overrides the defaults of three other environment variables, setting them to the values that a TLS-encrypted connection to LogStream Cloud requires. That results in these (and [other](/docs/logstream-integration#parameter-overrides))settings:

* `SCOPE_CRIBL_TLS_ENABLE` is set to `true`.
* `SCOPE_CRIBL_TLS_VALIDATE_SERVER` is set to `true`.
* `SCOPE_CRIBL_TLS_CA_CERT_PATH` is set to the empty string.

What if you want to route data to the cloud over an **unencrypted** connection? In that case, use `SCOPE_CRIBL`, as described [here](/docs/logstream-integration#connecting-to-logstream-cloud-unencrypted).

#### Routing with the Config File

TBD

### Routing Events

TBD

### Routing Metrics

TBD

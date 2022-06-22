---
title: Data Routing
---

## Data Routing

AppScope gives you multiple ways to route data. The basic operations are:

- Routing [both events and metrics](#routing-to-edge) to [Cribl Edge](https://docs.cribl.io/edge/).
- Routing [both events and metrics](#routing-to-cloud) to [Cribl Stream](https://docs.cribl.io/stream).
- Routing [events](#routing-events) to a file, local unix socket, network destination, or Cribl Edge.
- Routing [metrics](#routing-metrics) to a file, local unix socket, network destination, or Cribl Edge.
- [Muting](#muting) events, metrics, and payloads so that AppScope produces no output.

For each of these operations, the CLI has command-line options, the config file has settings, and the AppScope library has environment variables.

If you plan to use the config file, do take time to [read it all the way through](/docs/config-file) - then this page will make more sense!

### Routing to Cribl Edge {#routing-to-edge}

In a single operation, you can route both events and metrics to Cribl Edge. 

#### Routing to Edge with the CLI

Use the `-c` or `--cribldest` option. You need this option because, by default, the CLI routes data to the local filesystem. For example:

```
scope run -c edge -- curl https://wttr.in/94105
```

This usage works with `scope run`, `scope attach`, `scope watch`, `scope k8s`, and `scope service`. It does **not** work with `scope metrics`, where the `-c` option stands for `--cols`, telling the CLI to output data in columns instead of rows.

#### Routing to Edge with the AppScope Library 

Use the `SCOPE_CRIBL` environment variable to define a unix socket connection to Cribl Edge. For example:

```
LD_PRELOAD=./libscope.so SCOPE_CRIBL=edge ls -al
```

#### Routing to Edge with the Config File

Complete these steps, paying particular attention to the sub-elements of `cribl > transport`, which is where you specify routing:

* Verify that `cribl > enable` is set to `true`, which enables the `cribl` backend. (This is the default setting.)
* To route data to Cribl Edge, set `cribl > transport > type` to `edge`. (This is the default value.)

### Routing to Cribl Stream in the Cloud {#routing-to-cloud}

In a single operation, you can route both events and metrics to Cribl Stream in the Cloud, or Cribl.Cloud for short.

#### Routing to Cribl.Cloud with the CLI

Use the `-c` or `--cribldest` option. You need this option because, by default, the CLI routes data to the local filesystem. For example:

```
scope run -c tls://127.0.0.1:10090 -- curl https://wttr.in/94105
```

This usage works with `scope run`, `scope attach`, `scope watch`, `scope k8s`, and `scope service`. It does **not** work with `scope metrics`, where the `-c` option stands for `--cols`, telling the CLI to output data in columns instead of rows.

#### Routing to Cribl.Cloud with the AppScope Library 

Use the `SCOPE_CRIBL_CLOUD` environment variable to define a TLS-encrypted connection to Cribl.Cloud. Specify a transport type, a host name or IPv4 address, and a port number, for example:

```
LD_PRELOAD=./libscope.so SCOPE_CRIBL_CLOUD=tcp://in.logstream.<cloud_instance>.cribl.cloud:10090 ls -al
```

As a convenience, when you set `SCOPE_CRIBL_CLOUD`, AppScope automatically overrides the defaults of three other environment variables, setting them to the values that Cribl.Cloud via TLS requires. That produces these (and a few [other](/docs/cribl-integration#parameter-overrides)) settings:

* `SCOPE_CRIBL_TLS_ENABLE` is set to `true`.
* `SCOPE_CRIBL_TLS_VALIDATE_SERVER` is set to `true`.
* `SCOPE_CRIBL_TLS_CA_CERT_PATH` is set to the empty string.

If you prefer an **unencrypted** connection to Cribl.Cloud, use `SCOPE_CRIBL`, as described [here](/docs/cribl-integration#cloud-unencrypted).

#### Routing to Cribl.Cloud with the Config File

Complete these steps, paying particular attention to the sub-elements of `cribl > transport`, which is where you specify routing:

* Verify that `cribl > enable` is set to `true`, which enables the `cribl` backend. (This is the default setting.)
* To route data to Cribl.Cloud, set `cribl > transport > type` to `tcp`. (The default value is `edge`.)
* Specify desired values for the rest of the `cribl > transport` sub-elements, namely `host`, `port`, and `tls`.

### Routing Events {#routing-events}

You can route events independently of metrics.

#### Routing Events with the CLI

Use the `-e` or `--eventdest` option. For example:

```
scope run -e tcp://localhost:9109 -- curl https://wttr.in/94105
```

The above example sends events in [ndjson](http://ndjson.org/), the CLI's default output format. To send events in StatsD, you would include `--metricformat statsd` in the command.

#### Routing Events with the AppScope Library 

Use the `SCOPE_EVENT_DEST` environment variable, and set the `SCOPE_CRIBL_ENABLE` to `false`.

For example:

```
LD_PRELOAD=./libscope.so SCOPE_CRIBL_ENABLE=false SCOPE_EVENT_DEST=tcp://localhost:9109 curl https://wttr.in/94105
```

The above example sends events in StatsD, the AppScope library's default output format.

#### Routing Events with the Config File

Complete these steps, paying particular attention to the sub-elements of `event > transport`, which is where you specify routing:

* Set `cribl > enable` to `false` to disable the `cribl` backend.
* Set `event > enable` to `true` to enable the events backend.
* Specify desired values for the rest of the `event` elements, namely `format`, `watch`, and `transport`.

### Routing Metrics {#routing-metrics}

You can route metrics independently of events.

#### Routing Metrics with the CLI

Use the `-m` or `--metricdest` option. For example:

```
scope run -m udp://localhost:8125 -- curl https://wttr.in/94105
```

The above example sends events in [ndjson](http://ndjson.org/), the CLI's default output format. To send events in StatsD, you would include `--metricformat statsd` in the command.

#### Routing Events with the AppScope Library 

Use the `SCOPE_METRIC_DEST` environment variable, and set the `SCOPE_CRIBL_ENABLE` to `false`.

For example:

```
LD_PRELOAD=./libscope.so SCOPE_CRIBL_ENABLE=false SCOPE_METRIC_DEST=udp://localhost:8125 curl https://wttr.in/94105
```

The above example sends events in StatsD, the AppScope library's default output format.

This sends metrics in StatsD format. Adding `SCOPE_METRIC_FORMAT=ndjson` would change the format to ndjson.

#### Routing Metrics with the Config File

Complete these steps, paying particular attention to the sub-elements of `metric > transport`, which is where you specify routing:

* Set `cribl > enable` to `false` to disable the `cribl` backend.
* Set `metric > enable` to `true` to enable the metrics backend.
* Specify desired values for the rest of the `metric` elements, namely `format`, `transport`, and optionally, `watch`.

### Muting Events and Metrics {#muting}

You can use environment variables to turn off all event and metric output, as well as payloads. Set these variables to `false` to "mute" AppScope:

* `SCOPE_EVENT_ENABLE`
* `SCOPE_METRIC_ENABLE`
* `SCOPE_PAYLOAD_ENABLE`

Another approach is to create a custom config that omits (or comments out) all watch types in the `metric > watch[*]` and `event > watch[*]` arrays in the [config file](config-file). (Of course, you can turn any class of metric or event data on or off [individually](working-with#events-and-metrics), too.)

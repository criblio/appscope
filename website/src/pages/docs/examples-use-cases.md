---
title: Further Examples
---

## Further Examples

These examples provide details for most of the [use cases](what-do-with-scope#use-cases) introduced earlier.

#### Viewing Browser Results in a Dashboard

Run Firefox from the AppScope CLI, and view results on a terminal-based dashboard:

```
scope firefox
scope dash
```

#### Sending Metrics to Datadog

1. Send default metrics from the Go static application `hello` to the Datadog server at `ddog`:

```
scope run -m udp://ddog:8125
```

2. Send metrics from nginx to Datadog at `ddoghost`, using an environment variable and all other defaults:

```
LD_PRELOAD=/opt/appscope/libscope.so SCOPE_METRIC_DEST=udp://ddoghost:8125 nginx 
```

#### Sending Events from cURL

1. Send DNS events from cURL to the default host. This example uses the library in the current directory, independently of the CLI:

```
SCOPE_EVENT_DNS=true LD_PRELOAD=./libscope.so curl https://cribl.io
```

2. Send configured events from cURL to the server `myHost.example.com` on port 9000, using the config file at `/opt/appscope/scope.yml`:

```
scope run -u /opt/appscope/scope.yml -- curl https://cribl.io
```

Example config file:

```
event:
  enable: true
  transport:
    type: tcp
    host: myHost.example.com
    port: 9000
```

#### Sending HTTP Events to Splunk

Send HTTP events from Slack to a Splunk server at `myHost.example.com`:

```
scope run --passthrough -- slack
```

Example config file:

```
event:
  enable: true
  transport:
    type: tcp
    host: myHost.example.com
    port: 8083
...
  watch:
    - type: http
      name: .*
      field: .*
      value: .*
```

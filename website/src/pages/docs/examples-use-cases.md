---
title: Examples and Use Cases
---

## Examples and Use Cases

Here are some examples of using AppScope to monitor specific applications, and to send the resulting data to specific destinations.

#### Example 1: 
Run Firefox from the AppScope CLI, and view results on a terminal-based dashboard:

```
scope firefox
scope dash
```

#### Example 2
Send metrics from nginx to Datadog at `ddoghost`, using an environment variable and all other defaults:

```
SCOPE_METRIC_DEST=udp://ddoghost:5000 ldscope nginx
```

#### Example 3: 
Send configured events from curl to the server `myHost.example.com` on port 9000, using the config file at `/opt/scope/assets/scope.yml`:

```
SCOPE_HOME=/opt/scope/assets ldscope curl https://cribl.io
```

Configuration file example:

```
event:
  enable: true
  transport:
    type: tcp
    host: myHost.example.com
    port: 9000
```

#### Example 4: 
Send HTTP events from Slack to a Splunk server at `myHost.example.com`:

```
ldscope slack
```

Configuration file example, located at `/etc/scope` or `~/scope.yml`:

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

#### Example 5:
Send DNS events from curl to the default host. This example uses the library in the current directory, independently of the CLI or loader:

```
SCOPE_EVENT_DNS=true LD_PRELOAD=./libscope.so curl https://cribl.io
```

#### Example 6: 
Send default metrics from the Go static application `hello` to the Datadog server at `ddog`:

```
ldscope ./hello
```

Configuration file example:

```
metric:
  enable: true
  format:
    type : statsd
...
  transport:
    type: udp
    host: ddog
    port: 5000
```

### Next Steps

Explore [Integrating with LogStream](/docs/logstream-integration).
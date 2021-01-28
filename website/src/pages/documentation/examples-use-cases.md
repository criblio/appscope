# Examples and Use Cases

Here are some examples of using AppScope to monitor specific applications, and to send the resulting data to specific destinations.

1. Run Firefox from the AppScope CLI, and view results on a terminal-based dashboard:

```
scope firefox &
scope dash
```

2. Send metrics from nginx to Datadog at `ddoghost`, using an environment variable and all other defaults:

```
SCOPE_METRIC_DEST=udp://ddoghost:5000 ldscope nginx 
```

3. Send configured events from curl to the server `mydata` on port 9000, using the config file at `/opt/scope/scope.yml`:

```
SCOPE_HOME=/opt/scope ldscope curl https://cribl.io
```

Configuration file example:

```
event:
  enable: true
  transport:
    type: tcp 
    host: mydata
    port: 9000
```

4. Send HTTP events from Slack to a Splunk server at `shost`: 

```
ldscope slack
```

Configuration file example, located at `/etc/scope` or `~/scope.yml`:

```
event:
  enable: true
  transport:
    type: tcp
    host: shost
    port: 8088
-----
    - type: http
      name: .*
      field: .*
      value: .*
```

5. Send DNS events from curl to the default host. This example uses the library in the current directory, independently of the CLI or loader:

```
SCOPE_EVENT_DNS=true LD_PRELOAD=./libscope.so curl [https://cribl.io](https://cribl.io)
```

6. Send default metrics from the Go static application `hello` to the Datadog server at `ddog`:

```
ldscope ./hello 
```

Configuration file example:

```
metric:
  enable: true
  format:
    type : statsd
-----
transport:
    type: udp
    host: ddog
    port: 5000
```

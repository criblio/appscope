---
title: "Rules File"
---

## Rules File

The `scope start` command requires a Rules file, whose purpose is to specify processes to scope along with configs to apply to those processes.

The AppScope repo provides both a JSON and a YAML version of an example Rules file in the `conf` directory. The YAML version is reproduced below.

In Cribl Edge and Cribl Scope, instead of editing files, you create Rules in the **Process Monitoring** tab of the AppScope Source UI.

### How Rules Files Work

The Rules file begins with a top-level `allow` heading. Each Rule must provide a value for either the `procname` or the `arg` element, and leave the other element empty.

- Use `procname` to match an actual process name, e.g., `redis-server`.
- Use `arg` to perform a substring match on the entire command that starts a process. For example, `redis` will match all processes whose starting command includes `redis`, even if the process name itself does not.

After the `procname` and `arg` element comes the `config` element, which contains the same elements as a freestanding AppScope [config file](config-file). 

When a process matches a Rule, AppScope applies the specified `config` when scoping that process.

### Example Rules File

Here are the contents of `example_rules.yml`:

```
allow:
- procname: nginx
  arg:
  config:
    metric:
      enable: true
      format:
        statsdmaxlen: 512
        statsdprefix: null
        type: statsd
        verbosity: 4
      watch:
      - type: statsd
      - type: fs
      - type: net
      - type: http
      - type: dns
      - type: process
      transport:
        type: udp
        host: 127.0.0.1
        port: 8125
        tls:
          cacertpath: ''
          enable: false
          validateserver: true
    event:
      enable: true
      format:
        enhancefs: true
        maxeventpersec: 10000
        type: ndjson
      watch:
      - type: file
        name: (\/logs?\/)|(\.log$)|(\.log[.\d])
        value: .*
      - type: console
        name: (stdout)|(stderr)
        value: .*
        allowbinary: true
      - type: net
        name: .*
        field: .*
        value: .*
      - type: fs
        name: .*
        field: .*
        value: .*
      - type: dns
        name: .*
        field: .*
        value: .*
      - type: http
        name: .*
        field: .*
        value: .*
        headers:
      transport:
        type: tcp
        host: 127.0.0.1
        port: 9109
        tls:
          cacertpath: ''
          enable: false
          validateserver: true
    payload:
      dir: /tmp
      enable: false
    libscope:
      configevent: true
      summaryperiod: 10
      commanddir: /tmp
      log:
        level: warning
        transport:
          buffer: line
          path: /tmp/scope.log
          type: file
    cribl:
      enable: true
      transport:
        type: edge
        host: 127.0.0.1
        port: 10090
        tls:
          cacertpath: ''
          enable: false
          validateserver: true
- procname:
  arg: redis-server
  config:
    metric:
      enable: true
      format:
        statsdmaxlen: 512
        statsdprefix: null
        type: statsd
        verbosity: 4
      watch:
      - type: statsd
      - type: fs
      - type: net
      - type: http
      - type: dns
      - type: process
      transport:
        type: udp
        host: 127.0.0.1
        port: 8125
        tls:
          cacertpath: ''
          enable: false
          validateserver: true
    event:
      enable: true
      format:
        enhancefs: true
        maxeventpersec: 10000
        type: ndjson
      watch:
      - type: file
        name: (\/logs?\/)|(\.log$)|(\.log[.\d])
        value: .*
      - type: console
        name: (stdout)|(stderr)
        value: .*
        allowbinary: true
      - type: net
        name: .*
        field: .*
        value: .*
      - type: fs
        name: .*
        field: .*
        value: .*
      - type: dns
        name: .*
        field: .*
        value: .*
      - type: http
        name: .*
        field: .*
        value: .*
        headers:
      transport:
        type: tcp
        host: 127.0.0.1
        port: 9109
        tls:
          cacertpath: ''
          enable: false
          validateserver: true
    payload:
      dir: /tmp
      enable: false
    libscope:
      configevent: true
      summaryperiod: 10
      commanddir: /tmp
      log:
        level: warning
        transport:
          buffer: line
          path: /tmp/scope2.log
          type: file
    cribl:
      enable: true
      transport:
        type: edge
        host: 127.0.0.1
        port: 10090
        tls:
          cacertpath: ''
          enable: false
          validateserver: true
```

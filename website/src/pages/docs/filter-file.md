---
title: "Filter File"
---

## Filter File

The `scope start` command requires a filter file, whose purpose is to specify processes to scope along with configs to apply to those processes, and, optionally, processes **not** to scope.

The AppScope repo provides an example filter file in the `conf` directory, and reproduced below.

In Cribl Edge and Cribl Scope, instead of editing files, you create filters in the AppScope Source UI.

### Example Filter File

Here are the contents of `example_filter.yml`:

```
allow:
- procname: nginx
  arg: 
  config:
    cribl:
      enable: false
      transport:
        host: 127.0.0.1
        port: 10090
        tls:
          cacertpath: ''
          enable: false
          validateserver: true
        type: edge
    event:
      enable: true
      format:
        enhancefs: true
        maxeventpersec: 10000
        type: ndjson
      transport:
        host: 127.0.0.1
        port: 9109
        tls:
          cacertpath: ''
          enable: false
          validateserver: true
        type: tcp
      watch:
      - name: (\/logs?\/)|(\.log$)|(\.log[.\d])
        type: file
        value: .*
      - allowbinary: true
        name: (stdout)|(stderr)
        type: console
        value: .*
      - field: .*
        name: .*
        type: net
        value: .*
      - field: .*
        name: .*
        type: fs
        value: .*
      - field: .*
        name: .*
        type: dns
        value: .*
      - field: .*
        headers: null
        name: .*
        type: http
        value: .*
    libscope:
      commanddir: /tmp
      configevent: true
      log:
        level: warning
        transport:
          buffer: line
          path: /tmp/scope.log
          type: file
      summaryperiod: 10
    metric:
      enable: true
      format:
        statsdmaxlen: 512
        statsdprefix: null
        type: statsd
        verbosity: 4
      transport:
        host: 127.0.0.1
        port: 8125
        tls:
          cacertpath: ''
          enable: false
          validateserver: true
        type: udp
      watch:
      - type: statsd
      - type: fs
      - type: net
      - type: http
      - type: dns
      - type: process
    payload:
      dir: /tmp
      enable: false
- procname: redis
  arg: redis-server
  config:
    cribl:
      enable: true
      transport:
        host: 127.0.0.1
        port: 10090
        tls:
          cacertpath: ''
          enable: false
          validateserver: true
        type: edge
    event:
      enable: true
      format:
        enhancefs: true
        maxeventpersec: 10000
        type: ndjson
      transport:
        host: 127.0.0.1
        port: 9109
        tls:
          cacertpath: ''
          enable: false
          validateserver: true
        type: tcp
      watch:
      - name: (\/logs?\/)|(\.log$)|(\.log[.\d])
        type: file
        value: .*
      - allowbinary: true
        name: (stdout)|(stderr)
        type: console
        value: .*
      - field: .*
        name: .*
        type: net
        value: .*
      - field: .*
        name: .*
        type: fs
        value: .*
      - field: .*
        name: .*
        type: dns
        value: .*
      - field: .*
        headers: null
        name: .*
        type: http
        value: .*
    libscope:
      commanddir: /tmp
      configevent: true
      log:
        level: warning
        transport:
          buffer: line
          path: /tmp/scope2.log
          type: file
      summaryperiod: 10
    metric:
      enable: true
      format:
        statsdmaxlen: 512
        statsdprefix: null
        type: statsd
        verbosity: 4
      transport:
        host: 127.0.0.1
        port: 8125
        tls:
          cacertpath: ''
          enable: false
          validateserver: true
        type: udp
      watch:
      - type: statsd
      - type: fs
      - type: net
      - type: http
      - type: dns
      - type: process
    payload:
      dir: /tmp
      enable: false
deny:
- procname: git
```

# Snapshot information

TODO describe idea
TODO describe what information in which format files contain
TODO this is an interface description between CLI and library

Snapshot file details
```
/tmp/appscope/<PID>
│   info          <mandatory>

Example content:
```
Scope Version: v1.2.2-328-g73a7ded1e726
Unix Time: 1675350952 sec
PID: 1370399
Process name: htop
```

│   cfg           <mandatory>

Example content:
```
{"metric":{"enable":"true","transport":{"type":"file","path":"/home/testuser/.scope/history/htop_4_1370399_1675350868661357625/metrics.json","buffering":"line"},"format":{"type":"ndjson","statsdprefix":"","statsdmaxlen":512,"verbosity":4},"watch":[{"type":"fs"},{"type":"net"},{"type":"http"},{"type":"dns"},{"type":"process"},{"type":"statsd"}]},"libscope":{"log":{"level":"warning","transport":{"type":"file","path":"/home/testuser/.scope/history/htop_4_1370399_1675350868661357625/libscope.log","buffering":"line"}},"debug":{"stacktrace":"false"},"configevent":"false","summaryperiod":10,"commanddir":"/home/testuser/.scope/history/htop_4_1370399_1675350868661357625/cmd"},"event":{"enable":"true","transport":{"type":"file","path":"/home/testuser/.scope/history/htop_4_1370399_1675350868661357625/events.json","buffering":"line"},"format":{"type":"ndjson","maxeventpersec":10000,"enhancefs":"true"},"watch":[{"type":"file","name":"(\\/logs?\\/)|(\\.log$)|(\\.log[.\\d])","field":".*","value":".*"},{"type":"console","name":"(stdout|stderr)","field":".*","value":".*","allowbinary":"false"},{"type":"http","name":".*","field":".*","value":".*","headers":[]},{"type":"net","name":".*","field":".*","value":".*"},{"type":"fs","name":".*","field":".*","value":".*"},{"type":"dns","name":".*","field":".*","value":".*"}]},"payload":{"enable":"false","dir":"/tmp"},"tags":{},"protocol":[],"cribl":{"enable":"false","transport":{"type":"edge"},"authtoken":""}}
```
│   coredump      <optional>
|   backtrace     <optional>    
```
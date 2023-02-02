# Snapshot information

Snapshot feature is enabled by one of the following

- enablign the coredump, see `snapshot` -> `coredump` option in `scope.yml`
- enabling the backtrace, see `snapshot` -> `backtrace` option in `scope.yml`

Snapshot file details
The files will be placed in folllowing locations:
`/tmp/appscope/<PID>`

`/tmp/appscope/<PID>info`
This file is generated always when snapshot feature is enabled.

Example content:
```
Scope Version: v1.2.2-328-g73a7ded1e726
Unix Time: 1675350952 sec
PID: 1370399
Process name: htop
```

`/tmp/appscope/<PID>cfg`
This file is generated always when snapshot feature is enabled.

Example content:
```
{"metric":{"enable":"true","transport":{"type":"file","path":"/home/testuser/.scope/history/htop_4_1370399_1675350868661357625/metrics.json","buffering":"line"},"format":{"type":"ndjson","statsdprefix":"","statsdmaxlen":512,"verbosity":4},"watch":[{"type":"fs"},{"type":"net"},{"type":"http"},{"type":"dns"},{"type":"process"},{"type":"statsd"}]},"libscope":{"log":{"level":"warning","transport":{"type":"file","path":"/home/testuser/.scope/history/htop_4_1370399_1675350868661357625/libscope.log","buffering":"line"}},"snapshot":{"backtrace":"false"},"configevent":"false","summaryperiod":10,"commanddir":"/home/testuser/.scope/history/htop_4_1370399_1675350868661357625/cmd"},"event":{"enable":"true","transport":{"type":"file","path":"/home/testuser/.scope/history/htop_4_1370399_1675350868661357625/events.json","buffering":"line"},"format":{"type":"ndjson","maxeventpersec":10000,"enhancefs":"true"},"watch":[{"type":"file","name":"(\\/logs?\\/)|(\\.log$)|(\\.log[.\\d])","field":".*","value":".*"},{"type":"console","name":"(stdout|stderr)","field":".*","value":".*","allowbinary":"false"},{"type":"http","name":".*","field":".*","value":".*","headers":[]},{"type":"net","name":".*","field":".*","value":".*"},{"type":"fs","name":".*","field":".*","value":".*"},{"type":"dns","name":".*","field":".*","value":".*"}]},"payload":{"enable":"false","dir":"/tmp"},"tags":{},"protocol":[],"cribl":{"enable":"false","transport":{"type":"edge"},"authtoken":""}}
```

`/tmp/appscope/<PID>coredump`
This file is generated when coredump feature is enabled.
Contains the coredump generated during crash.
You can use it during further investigation with [gdb](https://sourceware.org/gdb/onlinedocs/gdb/Core-File-Generation.html)

`/tmp/appscope/<PID>backtrace`
This file is generated when backtrace feature is enabled.
Contains the backtrace captured during crash.

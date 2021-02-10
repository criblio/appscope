---
title: Quick Start Guide
---

## Quick Start Guide
---

### Explore the CLI

Run `scope --help` or `scope -h` to view CLI options. Also see the complete [CLI Reference](/docs/cli-reference).

### Let's scope some commands/applications

Test-run a well-known Linux commands, and view the results. E.g.:

```
scope ps -ef
>> should see output of ps here <<
```

Let's try another:

```
scope curl https://google.com
>> should see output of curl here <<
```


### Let's explore captured data

To see the monitoring and visualization features AppScope offers, exercise some of its options. E.g.:

- Show last session's captured metrics with `scope metrics`:

```
NAME         	VALUE	TYPE 	UNIT       	PID	TAGS
proc.start   	1    	Count	process    	525	args: %2Fusr%2Fbin%2Fps%20-ef,host: 771f60292e26,proc: ps
proc.cpu     	10000	Count	microsecond	525	host: 771f60292e26,proc: ps
proc.cpu_perc	0.1  	Gauge	percent    	525	host: 771f60292e26,proc: ps
proc.mem     	16952	Gauge	kibibyte   	525	host: 771f60292e26,proc: ps
proc.thread  	1    	Gauge	thread     	525	host: 771f60292e26,proc: ps
proc.fd      	6    	Gauge	file       	525	host: 771f60292e26,proc: ps
proc.child   	0    	Gauge	process    	525	host: 771f60292e26,proc: ps
fs.read      	77048	Count	byte       	525	class: summary,host: 771f60292e26,proc: ps
fs.seek      	2    	Count	operation  	525	class: summary,host: 771f60292e26,proc: ps
fs.stat      	15   	Count	operation  	525	class: summary,host: 771f60292e26,proc: ps
fs.open      	31   	Count	operation  	525	class: summary,host: 771f60292e26,proc: ps
fs.close     	29   	Count	operation  	525	class: summary,host: 771f60292e26,proc: ps
fs.duration  	83   	Count	microsecond	525	class: summary,host: 771f60292e26,proc: ps
fs.error     	7    	Count	operation  	525	class: stat,file: summary,host: 771f60292e26,op: summary,proc: ps
```


- Plot a chart of last session's `proc.cpu` metric with `scope metrics -m proc.cpu -g`:

```
 80000 ┼                   ╭╮
 76000 ┤                   ││
 72000 ┤                   ││
 68000 ┤                   ││
 64000 ┤                   ││
 60000 ┤                   ││
 56000 ┤                   ││
 52000 ┤                   ││                                     ╭╮                               ╭╮   ╭─
 48000 ┤                   ││                                     ││                               ││   │
 44000 ┤                   ││                                     ││                               ││   │
 40000 ┤ ╭╮                ││                                     ││          ╭╮                ╭─╮││ ╭╮│
 36000 ┤ ││                ││                                     ││          ││                │ │││ │││
 32000 ┤ ││╭╮              ││          ╭╮    ╭╮ ╭╮                ││ ╭╮       ││ ╭╮ ╭╮ ╭╮ ╭╮ ╭╮╭╯ ╰╯╰╮│╰╯
 28000 ┤ ││││              ││          ││    ││ ││                ││ ││       ││ ││ ││ ││ ││ │││     ││
 24000 ┤ ││││              ││          ││    ││ ││                ││ ││       ││ ││ ││ ││ ││ │││     ││
 20000 ┤╭╯││╰╮ ╭╮╭╮ ╭──╮╭╮ ││ ╭╮ ╭╮ ╭╮ ││ ╭──╯│ ││╭─╮ ╭╮    ╭───╮ ││╭╯│ ╭╮ ╭╮ │╰╮││ ││ ││ ││ │╰╯     ╰╯
 16000 ┤│ ││ │ ││││ │  │││ ││ ││ ││ ││ ││ │   │ │││ │ ││    │   │ │││ │ ││ ││ │ │││ ││ ││ ││ │
 12000 ┤│ ╰╯ ╰─╯╰╯╰─╯  ╰╯│╭╯╰─╯╰─╯│╭╯╰─╯╰─╯   ╰─╯╰╯ ╰─╯╰────╯   │╭╯╰╯ │╭╯╰─╯╰─╯ ╰╯╰─╯│╭╯╰─╯╰─╯
  8000 ┤│                ││       ││                            ││    ││             ││
  4000 ┤│                ││       ││                            ││    ││             ││
     0 ┼╯                ╰╯       ╰╯                            ╰╯    ╰╯             ╰╯
```



- Display last session's captured events with `scope events`:

```
[j98] Jan 31 21:38:53 ps console stdout 19:40
[2d8] Jan 31 21:38:53 ps console stdout pts/1
[Ng8] Jan 31 21:38:53 ps console stdout 00:00:11
[Ck8] Jan 31 21:38:53 ps console stdout /opt/cribl/bin/cribl /opt/cribl/bin/cribl.js server -r WORKER
[fp8] Jan 31 21:38:53 ps console stdout root
[Ys8] Jan 31 21:38:53 ps console stdout 518
[Lw8] Jan 31 21:38:53 ps console stdout 10
[uA8] Jan 31 21:38:53 ps console stdout 0
[aE8] Jan 31 21:38:53 ps console stdout 21:38
[VH8] Jan 31 21:38:53 ps console stdout pts/1
[EL8] Jan 31 21:38:53 ps console stdout 00:00:00
[tP8] Jan 31 21:38:53 ps console stdout ldscope ps -ef
[nT8] Jan 31 21:38:53 ps console stdout root
[4X8] Jan 31 21:38:53 ps console stdout 525
[T%8] Jan 31 21:38:53 ps console stdout 518
[C29] Jan 31 21:38:53 ps console stdout 0
[i69] Jan 31 21:38:53 ps console stdout 21:38
[1a9] Jan 31 21:38:53 ps console stdout pts/1
[Md9] Jan 31 21:38:53 ps console stdout 00:00:00
[Bh9] Jan 31 21:38:53 ps console stdout /usr/bin/ps -ef
```

- Filter out last session's events, for just `http` with `scope events -t http`:

```
[MJ33] Jan 31 19:55:22 cribl http http-resp http.host:localhost:9000 http.method:GET http.scheme:http http.target:/ http.response_content_length:1630
[RT33] Jan 31 19:55:22 cribl http http-metrics duration:0 req_per_sec:2
[Ta43] Jan 31 19:55:22 cribl http http-req http.host:localhost:9000 http.method:GET http.scheme:http http.target:/
[fj43] Jan 31 19:55:22 cribl http http-resp http.host:localhost:9000 http.method:GET http.scheme:http http.target:/ http.response_content_length:1630
[kt43] Jan 31 19:55:22 cribl http http-metrics duration:0 req_per_sec:2
[mM43] Jan 31 19:55:22 cribl http http-req http.host:localhost:9000 http.method:GET http.scheme:http http.target:/
[KU43] Jan 31 19:55:22 cribl http http-resp http.host:localhost:9000 http.method:GET http.scheme:http http.target:/ http.response_content_length:1630
[P253] Jan 31 19:55:22 cribl http http-metrics duration:0 req_per_sec:2
[Rl53] Jan 31 19:55:22 cribl http http-req http.host:localhost:9000 http.method:GET http.scheme:http http.target:/
[du53] Jan 31 19:55:22 cribl http http-resp http.host:localhost:9000 http.method:GET http.scheme:http http.target:/ http.response_content_length:1630
[iE53] Jan 31 19:55:22 cribl http http-metrics duration:0 req_per_sec:2
```


- List this AppScope sessions history with `scope history`:

```
Displaying last 20 sessions
ID	COMMAND	CMDLINE                  	PID	AGE   	DURATION	TOTAL EVENTS
1 	cribl  	/opt/cribl/bin/cribl sta…	50 	2h11m 	2h11m   	6275
2 	dahs   	dahs                     	109	2h11m 	0ms     	0
3 	curl   	curl https://google.com  	509	13m30s	206ms   	16
4 	ps     	ps -ef                   	518	13m18s	22ms    	120
```


### Next Steps

For guidance on taking AppScope to the next level, [join](https://cribl.io/community?utm_source=appscope&utm_medium=footer&utm_campaign=appscope) our [Community Slack](https://app.slack.com/client/TD0HGJPT5/CPYBPK65V/thread/C01BM8PU30V-1611001888.001100).

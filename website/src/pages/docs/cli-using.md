---
title: Using the CLI
---

## Using The Command Line Interface (CLI)

As soon as you download AppScope, you can start using the CLI to explore and gain insight into application behavior. No installation or configuration is required.

The CLI provides a rich set of capabilities for capturing and managing data from single applications. Data is captured in the local filesystem.

To learn more, see the [CLI Reference](/docs/cli-reference), and/or run `scope --help` or `scope -h`.

### CLI Basics

The basic AppScope CLI command is `scope`. The following examples provide an overview of how to use it.

**Scope a new process.** In these examples, we'll test-run well-known Linux commands and view the results:

```
scope top
```

```
scope ls -al
```

```
scope curl https://google.com
```

**Monitor an application that generates a large data set**.

```
scope firefox
```

You can run this and then try [exploring the captured data](#explore-captured).

**Scope a series of shell commands**. In the shell that you open in this example, every command you run will be scoped:

```
scope bash
```

To release a process like `scope bash` above, `exit` the process in the shell. If you do this with the bash shell itself, you won't be able to `scope bash` again in the same terminal session.


**Scope [Cribl LogStream](https://cribl.io/download/), if you have it installed**. As with other applications, it's interesting to see what AppScope shows LogStream doing:


`scope $CRIBL_HOME/bin/cribl server`

**Scope a running process.** This example scopes the process whose PID is `12345`:

```
sudo bin/linux/scope attach 12345
```

This example requires some preparation, as explained in [Scoping a Running Process](#scope-running).


**Invoke a configuration file while scoping a new process.** This example scopes the `echo` command while invoking a configuration file named `cloud.yml`:

```
scope run -u cloud.yml -- echo foo
```

Why create a custom configuration file? One popular reason is to change the destination to which AppScope sends captured data to someplace other than the local filesystem (the default). To learn about the possibilities, read the comments in the default [Configuration File](/docs/config-file).

<span id="scope-running"></span>

### Scoping a Running Process

You attach AppScope to a process using a process ID or a process name.

#### Attaching by Process ID

In this example, we grep for process IDs of `cribl` (LogStream) processes, then attach to a process of interest:

```
$ ps -ef | grep cribl
ubuntu    1820     1  1 21:03 pts/4    00:00:02 /home/ubuntu/someusername/cribl/3.1.2/m/cribl/bin/cribl server
ubuntu    1838  1820  4 21:03 pts/4    00:00:07 /home/ubuntu/someusername/cribl/3.1.2/m/cribl/bin/cribl /home/ubuntu/someusername/cribl/3.1.2/m/cribl/bin/cribl.js server -r CONFIG_HELPER
ubuntu    1925 30025  0 21:06 pts/3    00:00:00 grep --color=auto cribl
ubuntu@ip-127-0-0-1:~/someusername/appscope3$ sudo bin/linux/scope attach 1820

```

#### Attaching by Process Name

In this example, we try to attach to a LogStream process by its name, which will be `cribl`. Since there's more than one process, AppScope lists them and prompts us to choose one:

```
$ sudo bin/linux/scope attach cribl
Found multiple processes matching that name...
ID  PID   USER    SCOPED  COMMAND
1   1820  ubuntu  false   /home/ubuntu/someusername/cribl/3.1.2/m/cribl/bin/cribl server
2   1838  ubuntu  false   /home/ubuntu/someusername/cribl/3.1.2/m/cribl/bin/cribl /home/ubuntu/someusername/cribl/3.1.2/m/cribl/bin/cribl.js server -r CONFIG_HELPER
Select an ID from the list:
2
WARNING: Session history will be stored in /home/ubuntu/.scope/history and owned by root
Attaching to process 1838

```

#### More About Scoping Running Processes

To attach AppScope to a running process:

1. You must run `scope` as root, or with `sudo`.
1. If you attach to a shell, AppScope does not automatically scope its child processes.
1. You can attach to a process that is executing within a container context by running `scope attach` **inside** that container.
  - However:
    You **cannot** attach to a process that is executing within a container context by running `scope attach` **outside** that container (for example, in the host OS, or in a different container).

When you attach AppScope to a process, its child processes are not automatically scoped.

You cannot attach to a musl libc process or a static executable's process.

No HTTP/1.1 events and headers are emitted when AppScope attaches to a Go process that uses the `azure-sdk-for-go` package.

No events are emitted from files or sockets that exist before AppScope attaches to a process.

- After AppScope attaches to a process, whatever file descriptors and/or socket descriptors that the process had already opened before that,
AppScope will not report any **new** activity on those file or socket descriptors.

  - For example, suppose a process opens a socket descriptor before AppScope is attached. Subsequent sends and receives on this socket will not produce AppScope events.

   - AppScope events will be produced only for sockets opened **after** AppScope is attached.


<span id="explore-captured"></span>

### Exploring Captured Data

To see the monitoring and visualization features AppScope offers, exercise some of its sub-commands and options. E.g.:

- Show last session's captured metrics with `scope metrics`:

```
NAME         	VALUE	TYPE 	UNIT       	PID	TAGS
proc.start   	1    	Count	process    	525	args: %2Fusr%2Fbin%2Fps%20-ef,gid: 0,host: 771f60292e26,proc: ps,uid: 0
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



- Display the last session's captured events with `scope events`:

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

- Filter out everything but `http` from the last session's events, with `scope events -t http`:


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


- View the history of the current AppScope session with `scope history`:

```
Displaying last 20 sessions
ID	COMMAND	CMDLINE                  	PID	AGE   	DURATION	TOTAL EVENTS
1 	cribl  	/opt/cribl/bin/cribl sta…	50 	2h11m 	2h11m   	6275
2 	curl   	curl https://google.com  	509	13m30s	206ms   	16
3 	ps     	ps -ef                   	518	13m18s	22ms    	120
```

---
title: Using the CLI
---

## Using The Command Line Interface (CLI)

As soon as you download AppScope, you can start using the CLI to explore and gain insight into application behavior. No installation or configuration is required.

The CLI provides a rich set of capabilities for capturing and managing data from single applications. Data is captured in the local filesystem.

To learn more, see the [CLI Reference](/docs/cli-reference), and/or run `scope --help` or `scope -h`.

### CLI Basics

The basic AppScope CLI command is `scope`. The following examples provide an overview of how to use it.

#### Scope a New Process 

In these examples, we'll test-run well-known Linux commands and view the results:

```
scope top
```

```
scope ls -al
```

```
scope curl https://google.com
```

#### Monitor an App that Generates a Large Data Set

We use the `--no-sandbox` option because virtually all modern browsers incorporate the [sandbox](https://web.dev/browser-sandbox/) security mechanism, and some sandbox implementations can interact with AppScope in ways that cause the browser to hang, or not to start.

```
scope firefox --no-sandbox
```

After you scope the browser, try [exploring the captured data](#explore-captured).

#### Scope a Series of Shell Commands

In the shell that you open in this example, every command you run will be scoped:

```
scope bash
```

To release a process like `scope bash` above, `exit` the process in the shell. If you do this with the bash shell itself, you won't be able to `scope bash` again in the same terminal session.


#### Scope [Cribl Stream](https://cribl.io/download/)

As with other applications, it's interesting to see what AppScope shows Cribl Stream doing:

`scope $CRIBL_HOME/bin/cribl server`

#### Scope a Running Process

This example scopes the process whose PID is `12345`:

```
sudo bin/linux/scope attach 12345
```

This example requires some preparation, as explained in [Scoping a Running Process](#scope-running).


#### Invoke a Config File While Scoping a New Process

This example scopes the `echo` command while invoking a configuration file named `cloud.yml`:

```
scope run -u cloud.yml -- echo foo
```

Why create a custom configuration file? One popular reason is to change the destination to which AppScope sends captured data to someplace other than the local filesystem (the default). To learn about the possibilities, read the comments in the default [Configuration File](/docs/config-file).

#### Use a Custom HTTP Header to Mark Your Data

When scoping apps that make HTTP requests, you can add the `X-Appscope` header (setting its content to the string of your choice). 

Here's a trivial example of how this feature might be useful:

Suppose you want to scope HTTP requests to a weather forecast service, looking at many different cities, but labeling the data by country.

You could use the `X-Appscope` header to do your labeling:

```
$ ldscope curl --header "X-Appscope: Canada" wttr.in/calgary
$ ldscope curl --header "X-Appscope: Canada" wttr.in/ottawa
...

$ ldscope curl --header "X-Appscope: Brazil" wttr.in/recife
$ ldscope curl --header "X-Appscope: Brazil" wttr.in/riodejaneiro
...
```

In the resulting AppScope events, the `body` > `data` element would include an `"x-appscope"` field whose value would be the country named in the request's `X-Appscope` header.

<span id="scope-running"></span>

### Scoping a Running Process

You attach AppScope to a process using a process ID or a process name.

#### Attaching by Process ID

In this example, we grep for process IDs of `cribl` (Cribl Stream) processes, then attach to a process of interest:

```
$ ps -ef | grep cribl
ubuntu    1820     1  1 21:03 pts/4    00:00:02 /home/ubuntu/someusername/cribl/3.1.2/m/cribl/bin/cribl server
ubuntu    1838  1820  4 21:03 pts/4    00:00:07 /home/ubuntu/someusername/cribl/3.1.2/m/cribl/bin/cribl /home/ubuntu/someusername/cribl/3.1.2/m/cribl/bin/cribl.js server -r CONFIG_HELPER
ubuntu    1925 30025  0 21:06 pts/3    00:00:00 grep --color=auto cribl
ubuntu@ip-127-0-0-1:~/someusername/appscope3$ sudo bin/linux/scope attach 1820

```

#### Attaching by Process Name

In this example, we try to attach to a Cribl Stream process by its name, which will be `cribl`. Since there's more than one process, AppScope lists them and prompts us to choose one:

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

You cannot attach to a static executable's process.

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
proc.start   	1    	Count	process    	525	args: %2Fusr%2Fbin%2Fps%20-ef,gid: 0,groupname: root,host: 771f60292e26,proc: ps,uid: 0,username: root
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

- Note that by default, the AppScope CLI redacts binary data from console output. Although in most situations, the default behaviors of the AppScope CLI and library are the same, they differ for binary data: it's omitted in the CLI, and allowed when using the library. To change this, use the `allowbinary=true` flag. The equivalent environment variable is `SCOPE_ALLOW_BINARY_CONSOLE`. In the config file, `allowbinary` is an attribute of the `console` watch type for events.  

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
[MJ33] Jan 31 19:55:22 cribl http http.resp http.host:localhost:9000 http.method:GET http.scheme:http http.target:/ http.response_content_length:1630
[RT33] Jan 31 19:55:22 cribl http http-metrics duration:0 req_per_sec:2
[Ta43] Jan 31 19:55:22 cribl http http.req http.host:localhost:9000 http.method:GET http.scheme:http http.target:/
[fj43] Jan 31 19:55:22 cribl http http.resp http.host:localhost:9000 http.method:GET http.scheme:http http.target:/ http.response_content_length:1630
[kt43] Jan 31 19:55:22 cribl http http-metrics duration:0 req_per_sec:2
[mM43] Jan 31 19:55:22 cribl http http.req http.host:localhost:9000 http.method:GET http.scheme:http http.target:/
[KU43] Jan 31 19:55:22 cribl http http.resp http.host:localhost:9000 http.method:GET http.scheme:http http.target:/ http.response_content_length:1630
[P253] Jan 31 19:55:22 cribl http http-metrics duration:0 req_per_sec:2
[Rl53] Jan 31 19:55:22 cribl http http.req http.host:localhost:9000 http.method:GET http.scheme:http http.target:/
[du53] Jan 31 19:55:22 cribl http http.resp http.host:localhost:9000 http.method:GET http.scheme:http http.target:/ http.response_content_length:1630
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

- Scope a command which will produce a variety of events ...

```
# scope run -- apt update
Get:1 http://security.ubuntu.com/ubuntu focal-security InRelease [114 kB]
Get:2 http://archive.ubuntu.com/ubuntu focal InRelease [265 kB]
Get:3 http://security.ubuntu.com/ubuntu focal-security/restricted amd64 Packages [1027 kB]
...
```

- Then, show only the last 10 events of sourcetype `console`, sorted in ascending order by timestamp:

```
# scope events --sort _time --sourcetype console --last 10 --reverse
[IHI] Mar 15 16:22:15 apt console stdout message:" 0% [Working]              Get:1 http://security.ubuntu.com/ubuntu focal-security InRelease [114 kB]  0% [Waiting for …"
[jrL] Mar 15 16:22:15 http console stdout message:"102 Status URI: http://archive.ubuntu.com/ubuntu/dists/focal/InRelease Message: Connecting to archive.ubuntu.com  102…"
[OUJ] Mar 15 16:22:15 http console stdout message:"102 Status URI: http://security.ubuntu.com/ubuntu/dists/focal-security/InRelease Message: Connecting to security.ubun…"
[4nT] Mar 15 16:22:16 gpgv console stdout message:"201 URI Done GPGVOutput: GOODSIG 3B4FE6ACC0B21F32  GOODSIG 871920D1991BC93C Signed-By: 790BC7277767219C42C86F933B4FE6…"
[9WG] Mar 15 16:22:16 gpgv console stdout message:"201 URI Done GPGVOutput: GOODSIG 3B4FE6ACC0B21F32  GOODSIG 871920D1991BC93C Signed-By: 790BC7277767219C42C86F933B4FE6…"
[hYZ] Mar 15 16:22:16 store console stdout message:"200 URI Start URI: store:/var/lib/apt/lists/partial/security.ubuntu.com_ubuntu_dists_focal-security_restricted_binar…"
[PmG1] Mar 15 16:22:18 apt console stdout message:" 60% [10 Packages 6535 kB/11.3 MB 58%]                                       78% [Waiting for headers]               …"
[CVa1] Mar 15 16:22:18 apt console stdout message:" 60% [10 Packages 6535 kB/11.3 MB 58%]                                       78% [Waiting for headers]               …"
[yC41] Mar 15 16:22:18 http console stdout message:"201 URI Done SHA256-Hash: 200acdc3421757fa8f8759c1cedacae2cf8f5821d7810ca59d8a313c5b3ae71e SHA1-Hash: 28dce64e724849…"
[oZH1] Mar 15 16:22:20 apt console stdout message:"Building dependency tree         Reading state information... 0%  Reading state information... 48%  Reading state inf…"
```

- Scope a command which reads and writes over the network:
  
```
# scope curl wttr.in    
Weather report: San Francisco, California, United States

                Mist
   _ - _ - _ -  +55(53) °F     
    _ - _ - _   ↖ 10 mph       
   _ - _ - _ -  4 mi           
                0.0 in         
...
```
- Then, show only events containing the string `net_bytes`, and display the fields `net_bytes_sent` and `net_bytes_recv`, with their values:

```
scope events --fields net_bytes_sent,net_bytes_recv --match net_bytes
[XG] Mar 15 17:56:38 curl net net.close net_bytes_sent:1 net_bytes_recv:0
[zf1] Mar 15 17:56:38 curl net net.close net_bytes_sent:71 net_bytes_recv:8844
```

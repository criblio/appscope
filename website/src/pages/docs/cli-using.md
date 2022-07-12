---
title: Using the CLI
---

## Using The Command Line Interface (CLI)

As soon as you [download](/docs/downloading) AppScope, you can start using the CLI to explore and gain insight into application behavior. No installation or configuration is required.

The CLI provides a rich set of capabilities for capturing and managing data from single applications. Data is captured in the local filesystem.

By default, the AppScope CLI redacts binary data from console output. Although in most situations, the default behaviors of the AppScope CLI and library are the same, they differ for binary data: it's omitted in the CLI, and allowed when using the library. To change this, use the `allowbinary=true` flag. The equivalent environment variable is `SCOPE_ALLOW_BINARY_CONSOLE`. In the config file, `allowbinary` is an attribute of the `console` watch type for events. 

To learn more, see the [CLI Reference](/docs/cli-reference), and/or run `scope --help` or `scope -h`. And check out the [Further Examples](/docs/examples-use-cases), which include both CLI and library use cases.

### CLI Basics

The basic AppScope CLI command is `scope`. The following examples progress from simple to more involved. Scoping a running process gets [its own section](/docs/scope-running). The final section explains how to scope an app that generates a large data set, and then [explore](/docs/explore-captured) that data.

#### Scope a New Process 

Try scoping some well-known Linux commands, and view the results:

```
scope top
```

```
scope ls -al
```

```
scope curl https://www.google.com
```

When scoping browsers, we use the `--no-sandbox` option because virtually all modern browsers incorporate the [sandbox](https://web.dev/browser-sandbox/) security mechanism, and some sandbox implementations can interact with AppScope in ways that cause the browser to hang, or not to start.

```
scope firefox --no-sandbox
```

#### Scope a Series of Shell Commands

In the shell that you open in this example, every command you run will be scoped:

```
scope bash
```

To release a process like `scope bash` above, `exit` the process in the shell. If you do this with the bash shell itself, you won't be able to `scope bash` again in the same terminal session.

#### Scope [Cribl Stream](https://cribl.io/download/)

See what AppScope shows Cribl Stream doing:

`scope $CRIBL_HOME/bin/cribl server`

You can take the idea of using AppScope to reveal application behavior [much further](https://cribl.io/blog/appscope-1-0-changing-the-game-for-infosec-part-2/). 

#### Use a Custom HTTP Header to Mark Your Data

When scoping apps that make HTTP requests, you can add the `X-Appscope` header, setting its content to the string of your choice. 

Here's a trivial example of how this feature might be useful:

Suppose you want to scope HTTP requests to a weather forecast service, looking at many different cities, but labeling the data by country.

You could use the `X-Appscope` header to do your labeling:

```
scope curl --header "X-Appscope: Canada" wttr.in/calgary
scope curl --header "X-Appscope: Canada" wttr.in/ottawa
...

scope curl --header "X-Appscope: Brazil" wttr.in/recife
scope curl --header "X-Appscope: Brazil" wttr.in/riodejaneiro
...
```

In the resulting AppScope events, the `body` > `data` element would include an `"x-appscope"` field whose value would be the country named in the request's `X-Appscope` header.


#### Invoke a Config File While Scoping a New Process

Why create a custom configuration file? One popular reason is to change the destination to which AppScope sends captured data to someplace other than the local filesystem (the default). To learn about the possibilities, read the comments in the default [Configuration File](/docs/config-file).

This example scopes the `echo` command while invoking a configuration file named `cloud.yml`:

```
scope run -u cloud.yml -- echo foo
```

<span id="scope-running"></span>

### Scoping a Running Process

You attach AppScope to a process using either a process ID or a process name.

#### Attaching by Process ID

In this example, we'll start by grepping for process IDs of `cribl` (Cribl Stream) processes: 

```
$ ps -ef | grep cribl
ubuntu    1820     1  1 21:03 pts/4    00:00:02 /home/ubuntu/someusername/cribl/3.1.2/m/cribl/bin/cribl server
ubuntu    1838  1820  4 21:03 pts/4    00:00:07 /home/ubuntu/someusername/cribl/3.1.2/m/cribl/bin/cribl /home/ubuntu/someusername/cribl/3.1.2/m/cribl/bin/cribl.js server -r CONFIG_HELPER
ubuntu    1925 30025  0 21:06 pts/3    00:00:00 grep --color=auto cribl
```

Then, we'll attach to a process of interest:

```
ubuntu@ip-127-0-0-1:~/someusername/appscope3$ sudo scope attach 1820
```

#### Attaching by Process Name

In this example, we try to attach to a Cribl Stream process by its name, which will be `cribl`. Since there's more than one process, AppScope lists them and prompts us to choose one:

```
$ sudo scope attach cribl
Found multiple processes matching that name...
ID  PID   USER    SCOPED  COMMAND
1   1820  ubuntu  false   /home/ubuntu/someusername/cribl/3.5.1/m/cribl/bin/cribl server
2   1838  ubuntu  false   /home/ubuntu/someusername/cribl/3.5.1/m/cribl/bin/cribl /home/ubuntu/someusername/cribl/3.5.1/m/cribl/bin/cribl.js server -r CONFIG_HELPER
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

- After AppScope attaches to a process, AppScope will not report any **new** activity on file or socket descriptors that the process had already opened.

  - For example, suppose a process opens a socket descriptor before AppScope is attached. Subsequent sends and receives on this socket will not produce AppScope events.

   - AppScope events will be produced only for sockets opened **after** AppScope is attached.

<span id="explore-captured"></span>

### Exploring Captured Data

You can scope apps that generate large data sets, and then use AppScope CLI sub-commands and options to monitor and visualize the data.

Run the following command and then we'll explore how to monitor and visualize the results of the session:

```
scope run -- ps -ef
```

Show last session's captured metrics with `scope metrics`:

```
NAME             VALUE      TYPE     UNIT           PID     TAGS
proc.start       1          Count    process        5470    args: ps -ef,gid: 0,groupname: root,host: ip-10-8-107-15…
proc.cpu         60898      Count    microsecond    5470    host: my_hostname,proc: ps
proc.cpu_perc    0.60898    Gauge    percent        5470    host: my_hostname,proc: ps
proc.mem         67512      Gauge    kibibyte       5470    host: my_hostname,proc: ps
proc.thread      2          Gauge    thread         5470    host: my_hostname,proc: ps
proc.fd          6          Gauge    file           5470    host: my_hostname,proc: ps
proc.child       1          Gauge    process        5470    host: my_hostname,proc: ps
fs.read          379832     Count    byte           5470    host: my_hostname,proc: ps,summary: true
net.rx           401        Count    byte           5470    class: unix_tcp,host: my_hostname,proc: ps,summary: …
net.tx           361        Count    byte           5470    class: unix_tcp,host: my_hostname,proc: ps,summary: …
fs.seek          150        Count    operation      5470    host: my_hostname,proc: ps,summary: true
fs.stat          136        Count    operation      5470    host: my_hostname,proc: ps,summary: true
...
```

Plot a chart of last session's `proc.cpu` metric with `scope metrics -m proc.cpu -g`:

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

Display the last session's captured events with `scope events`:

```
[EkA1] Jul 12 02:11:15 ps fs fs.open file:/proc/17721/stat
[OrA1] Jul 12 02:11:15 ps fs fs.close file:/proc/17721/stat file_read_bytes:158 file_read_ops:1 file_write_bytes:0 file_write_ops:0
[xAA1] Jul 12 02:11:15 ps fs fs.open file:/proc/17721/status
[LHA1] Jul 12 02:11:15 ps fs fs.close file:/proc/17721/status file_read_bytes:952 file_read_ops:1 file_write_bytes:0 file_write_ops:0
[vQA1] Jul 12 02:11:15 ps fs fs.open file:/proc/17721/cmdline
[IXA1] Jul 12 02:11:15 ps fs fs.close file:/proc/17721/cmdline file_read_bytes:0 file_read_ops:1 file_write_bytes:0 file_write_ops:0
[s6B1] Jul 12 02:11:15 ps fs fs.open file:/proc/17758/stat
...
```

Now scope a command which produces HTTP and other kinds of events:

```
scope run -- curl https://www.google.com
```

Filter out everything but `http` from the last session's events, with `scope events -t http`:

```
[vL1] Jul 12 02:14:07 curl http http.req http_host:www.google.com http_method:GET http_scheme:https http_target:/
[PU1] Jul 12 02:14:08 curl http http.resp http_host:www.google.com http_method:GET http_target:/
```

View the history of the current AppScope session with `scope history`:

```
Displaying last 20 sessions
ID	COMMAND	CMDLINE                  	PID 	AGE 	DURATION	TOTAL EVENTS
4 	ps     	ps -ef                   	5464	3m6s	12ms    	788
5 	curl   	curl https://www.google.…	5492	14s 	141ms   	20
```

Scope another command which will produce a variety of events ...

```
# scope run -- apt update
Hit:1 http://us-west-2.ec2.archive.ubuntu.com/ubuntu bionic InRelease
Get:2 http://us-west-2.ec2.archive.ubuntu.com/ubuntu bionic-updates InRelease [88.7 kB]                                      
Get:3 http://us-west-2.ec2.archive.ubuntu.com/ubuntu bionic-backports InRelease [74.6 kB]                                    
Hit:4 https://download.docker.com/linux/ubuntu bionic InRelease                                                                                
Get:5 http://security.ubuntu.com/ubuntu bionic-security InRelease [88.7 kB]                                                                    
Hit:6 https://baltocdn.com/helm/stable/debian all InRelease
...
```

... Then, show only the last 10 events of sourcetype `console`, sorted in ascending order by timestamp:

```
# scope events --sort _time --sourcetype console --last 10 --reverse
[g2Wg] Jul 12 02:14:49 apt console stdout message:"                                  0% [Working] 0% [16 Packages store 0 B]                …"
[zvUe] Jul 12 02:14:49 apt console stdout message:"                                  0% [Working] 0% [16 Packages store 0 B]                …"
[ucRe] Jul 12 02:14:50 find console stdout message:/var/lib/apt/lists/security.ubuntu.com_ubuntu_dists_bionic-security_universe_binary-amd6…
[zkRe] Jul 12 02:14:50 dirname console stdout message:/var/lib/update-notifier
[I0Se] Jul 12 02:14:50 mktemp console stdout message:/var/lib/update-notifier/tmp.ut1sgTEuvb
[gmUg] Jul 12 02:14:54 python3 console stdout message:"39 updates can be applied immediately. To see these additional updates run: apt list …"
[gZ7h] Jul 12 02:14:54 apt console stdout message:"Reading package lists... 0%"
[1a2i] Jul 12 02:14:54 apt console stdout message:"Reading package lists... 0%  Reading package lists... 0%  Reading package lists... 3%  Re…"
[cp1i] Jul 12 02:14:54 apt console stdout message:"Reading package lists... 0%  Reading package lists... 0%  Reading package lists... 3%  Re…"
[QU2i] Jul 12 02:14:56 apt console stderr message:"W: An error occurred during the signature verification. The repository is not updated and…"
```

Scope a command which reads and writes over the network:
  
```
# scope curl wttr.in    
Weather report: Portland, Oregon, United States

      \   /     Sunny
       .-.      93 °F          
    ― (   ) ―   ↓ 6 mph        
       `-’      9 mi           
      /   \     0.0 in         
                                                       ┌─────────────┐                                                       
┌──────────────────────────────┬───────────────────────┤  Mon 11 Jul ├───────────────────────┬──────────────────────────────┐
│            Morning           │             Noon      └──────┬──────┘     Evening           │             Night            │
├──────────────────────────────┼──────────────────────────────┼──────────────────────────────┼──────────────────────────────┤
│     \   /     Sunny          │     \   /     Sunny          │     \   /     Sunny          │     \   /     Clear          │
│      .-.      68 °F          │      .-.      89 °F          │      .-.      +95(93) °F     │      .-.      78 °F          │
│   ― (   ) ―   ↓ 5-6 mph      │   ― (   ) ―   ↘ 6-7 mph      │   ― (   ) ―   ↘ 10-11 mph    │   ― (   ) ―   ↘ 8-15 mph     │
│      `-’      6 mi           │      `-’      6 mi           │      `-’      6 mi           │      `-’      6 mi           │
│     /   \     0.0 in | 0%    │     /   \     0.0 in | 0%    │     /   \     0.0 in | 0%    │     /   \     0.0 in | 0%    │
└──────────────────────────────┴──────────────────────────────┴──────────────────────────────┴──────────────────────────────┘       
...
```

Then, show only events containing the string `net_bytes`, and display the fields `net_bytes_sent` and `net_bytes_recv`, with their values:

```
# scope events --fields net_bytes_sent,net_bytes_recv --match net_bytes
[e91] Jul 12 02:15:41 curl net net.close net_bytes_sent:71 net_bytes_recv:8773
```

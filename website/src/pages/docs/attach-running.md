---
title: Scoping a Running Process
---

## Scoping a Running Process

You attach AppScope to a process using a process ID or a process name.

### Attaching by Process ID

In this example, we grep for process IDs of `cribl` (LogStream) processes, then attach to a process of interest:

```
$ ps -ef | grep cribl
ubuntu    1820     1  1 21:03 pts/4    00:00:02 /home/ubuntu/someusername/cribl/3.1.2/m/cribl/bin/cribl server
ubuntu    1838  1820  4 21:03 pts/4    00:00:07 /home/ubuntu/someusername/cribl/3.1.2/m/cribl/bin/cribl /home/ubuntu/someusername/cribl/3.1.2/m/cribl/bin/cribl.js server -r CONFIG_HELPER
ubuntu    1925 30025  0 21:06 pts/3    00:00:00 grep --color=auto cribl
ubuntu@ip-127-0-0-1:~/someusername/appscope3$ sudo bin/linux/scope attach 1820

```

### Attaching by Process Name

In this example, we try to attach to a LogStream process by its name, which will be `cribl`. Since there's more than one, AppScope lists them and prompts us to choose one:

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

## More About Scoping Running Processes

To attach AppScope to a running process:

1. You must run `scope` as root, or with `sudo`.
1. If you attach to a shell, AppScope does not automatically scope its child processes.
1. You can attach to a process that is executing within a container context by running `scope attach` **inside** that container.
  - However:
    You **cannot** attach to a process that is executing within a container context by running `scope attach` **outside** that container (for example, in the host OS, or in a different container).

When you attach AppScope to a process, its child processes are not automatically scoped.

You cannot attach to a musl libc process or a static executable's process.

No HTTP/1.1 events and headers are emitted when AppScope attaches to a Go process that uses the `azure-sdk-for-go` package.

No events are emitted from files or sockets that exists before AppScope attaches to a process.

- After AppScope attaches to a process, whatever file descriptors and/or socket descriptors that the process had already opened before that,
AppScope will not report any **new** activity on those file or socket descriptors.

  - For example, suppose a process opens a socket descriptor before AppScope is attached. Subsequent sends and receives on this socket will not produce AppScope events.

   - AppScope events will be produced only for sockets opened **after** AppScope is attached.

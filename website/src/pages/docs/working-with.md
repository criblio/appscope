---
title: Working With AppScope
---

## Working With AppScope

AppScope offers two ways to work:

* Use the CLI when you want to explore in real-time in an ad hoc way.

* Use the AppScope library (`libscope`) for longer-running, planned procedures. 

This principle should be taken more as a starting point than as a strict rule. You may sometimes want to plan out a CLI session, or (conversely) explore using the library.

What matters most is whether the command you want to scope can be changed while it is running. If it can, try the CLI; if not, go for the library.

For example:

* TBD example 1

* TBD example 2

* TBD example 3


TBD decision tree

### The Command Line Interface (CLI)

The AppScope CLI (`scope`) [provides](/docs/cli-using) a quick and convenient way to explore capabilities, and to obtain insight into application behavior. No installation or configuration is required to get started exploring application behavior.

This provides a rich set of capabilities intended to capture data from single applications. Data is captured in the local file system, and is managed through the CLI.

To scope a new process, for example the `top` program:

```
scope top
```

To [scope a running process](/docs/attach-running) whose process ID (PID) is `12345`:

```
scope attach 12345
```

To invoke a configuration file while scoping a new `some_example` process:

```
scope insert correct command here
```

TBD say something about the config file and what it does.

Flags that override a subset of env variables...


### The Library

The AppScope library (`libscope`) is the [core component](/docs/loader-library) that resides in application processes, extracting data as an application executes. You can closely configure the library's behavior using environment variables or a configuration [file](/docs/config-file). 

The library extracts information by interposing functions. When an application calls a function, it actually calls a function of the same name in `libscope`, which extracts details from the function. Then the original function call proceeds. This interposition of function calls requires no change to an application: it works with unmodified binaries, and its CPU and memory overhead is minimal.

The library is loaded using a number of mechanisms, depending on the type of executable. A dynamic loader can preload the library (where supported), and AppScope is used to load static executables.

Child processes are created with the library present, if the library was present in the parent. In this manner, a single executable is able to start, daemonize, and create any number of children, all of which include interposed functions.

You use an environment variable to load the library independent of any executable. Whether you use this option or the AppScope loader, you get full control of the data source, formats, and transports.

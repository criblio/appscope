---
title: Troubleshooting and Crash Analysis
---

You can use AppScope to [troubleshoot an application](#crash-analysis) that's behaving problematically:

- Whenever a scoped app crashes, AppScope can obtain a core dump, a backtrace (i.e., stack trace), or both, while capturing supplemental information in text files.

- AppScope can generate a **snapshot** file containing debug information about processes that are running normally or crashing, unscoped or scoped.

On the other hand, you can [troubleshoot AppScope itself](#troubleshoot-appscope) if you're trying to scope an application but not getting the results you expect:

- AppScope can retrieve the config currently in effect; you can then update (modify) it dynamically as desired.

- AppScope can determine the status of the transport it's trying to use to convey events and metrics to a destination. This helps troubleshoot "Why am I getting no events or metrics?" scenarios.

<span id="crash-analysis"></span>

## Crash Analysis

The `scope snapshot` command obtains debug information about a running or crashing process, regardless of whether or not the process is scoped.

Whenever the kernel sends the scoped app a fatal signal (i.e., illegal instruction, bus error, segmentation fault, or floating point exception), AppScope will capture either a coredump, a stacktrace, or both, depending on what you have configured or the command-line options you choose.

- See the `snapshot` section of the [config file](/docs/config-file).

- The `scope attach`, `scope run`, and `scope watch` commands can all take the `-b`/`backtrace` and `-c`/`--coredump` options.  

The `snapshot` [field](/docs/schema-reference/#eventstartmsginfoconfigurationcurrentlibscopesnapshot) in the process-start message records whether capturing coredumps and backtraces are enabled or not.

<span id="troubleshoot-appscope"></span>

## Troubleshooting AppScope itself

AppScope offers many options for sending data to other applications (Cribl Stream, Cribl Edge, or any other application that receives data over TCP or UNIX sockets). If data does not show up at the destination, you can use the CLI to troubleshoot the transport AppScope is trying to use:

- `scope inspect` retrieves the AppScope config currently in effect and determines the status of the transport AppScope is trying to use.
- `scope update` modifies the current AppScope config.
- `scope ps` now determines whether the processes it lists are scoped or not.

<!-- TBD talk about AppScope crashing -->

<!-- TBD fix links since this moved -->

<span id="dynamic-configuration"></span>

### Dynamic Configuration Via the Command Directory

Separately from the `scope update` command, AppScope offers another way to do dynamic configuration in real time, via the **command directory**.

To use this feature, you first need to know the PID (process ID) of the currently scoped process. You then create a `scope.<pid>` file, where:
- The `<pid>` part of the filename is the PID of the scoped process.
- Each line of the file consists of one configuration setting in the form `ENVIRONMENT_VARIABLE=value`.
  
Once per reporting period, AppScope looks in its **command directory** for a `scope.<pid>` file. If it finds one where the `<pid>` part of the filename matches the PID of the current scoped process, AppScope applies the configuration settings specified in the file, then deletes the file.

The command directory defaults to `/tmp` and the reporting period (a.k.a. metric summary interval) defaults to 10 seconds. You can configure them in the `libscope` section of the [config file](/docs/config-file)). See the `Command directory` and `Metric summary interval` subsections, respectively.

#### Dynamic configuration example

Suppose your process of interest has PID `4242` and to begin your investigation, you want to bump metric verbosity up to 9 while muting all HTTP events. 

Create a file named `scope.4242` that contains two lines:

```
SCOPE_METRIC_VERBOSITY=9
SCOPE_EVENT_HTTP=false
```

Save the `scope.4242`  file in the command directory. 

AppScope will detect the file and the matching process, apply the config settings, then delete the file.

When you've seen enough of the verbose metrics, and you want to start looking at HTTP events, create a new `scope.4242` which contains the following two lines, reversing the previous config settings:

```
SCOPE_METRIC_VERBOSITY=4
SCOPE_EVENT_HTTP=true
```

AppScope will detect the file and the matching process, apply the config settings, then delete the file.

You can iterate on this workflow depending on what you find and what more you want to discover.
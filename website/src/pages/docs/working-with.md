---
title: Working With AppScope
---

# Working With AppScope

There are three main things to know to work effectively with AppScope:

* Your [overall approach](#ad-hoc-vs-planned) can be either spontaneous or more planned-out.
* You can [control](#config-file-etc) AppScope by means of the config file, environment variables, or a combination of the two.
* The results you get from AppScope will be in the form of [events and metrics](#events-and-metrics).

## Ad Hoc Versus Planned-out {#ad-hoc-vs-planned}

AppScope offers two ways to work:

* Use the CLI when you want to explore in real time, in an ad hoc way.

* Use the AppScope library (`libscope`) for longer-running, planned procedures. 

This is a guiding principle, not a strict rule. Sometimes you may prefer to plan out a CLI session, or, conversely, explore using the library.

Ask yourself whether, as you work, you will vary the options or arguments to the command you want to scope. If you plan to make these kinds of iterative changes, try the CLI; if not, go for the library.

For example:

* You are developing code, and you want to see how its behavior changes as you iterate. **Use the CLI.**

* You are running cURL commands against a website, and want to see what changes when you switch between HTTP/1.1 and HTTP/2.2 and/or HTTP and HTTPS. **Use the CLI.** 

* You are running nginx in a specific, unchanging way dictated by the requirements of your organization, and you want to see metrics. **Use the library.**

Here's a decision tree to help you determine whether to use the CLI or the library.

![CLI vs. Library Decision Tree](./images/appscope_tree@2x.jpg)

## The Config File, Env Vars, Flags, and `ldscope` {#config-file-etc}

AppScope's ease of use stems from its flexible set of controls:

* AppScope's configuration file, `scope.yml`, can be invoked from either the CLI or the library.

* The AppScope library provides an extensive set of environment variables, which control settings like metric verbosity and event destinations. Environment variables override config file settings.

* Finally, AppScope provides the `ldscope` utility, whose uses include loading the AppScope library into Go executables.

Check out the [CLI](/docs/cli-using) and [library](/docs/library-using) pages to see how it's done.

## Events and Metrics in AppScope {#events-and-metrics}

When AppScope has [interposed a function](/docs/how-works), and then the application being scoped executes that function, AppScope can emit events and metrics. Both events and metrics follow patterns that are rigorously defined in validatable [JSON Schema](https://json-schema.org/), and documented in the [Schema Reference](/docs/schema-reference).

Events describe the action performed by the interposed function. For example, a `net.open` event could tell you that a network connection was established by Firefox, from the local IP/port `172.16.198.210:54388` to the remote IP/port `34.107.221.82:80`, using HTTP over TCP.

Metrics can do any of three things:
* When verbosity is set to lower values, metrics summarize aspects of the action performed by the interposed function, for a reporting period that defaults to 10 seconds. (See the `Metric verbosity level` section of the [config file](/docs/config-file).) For example, a `fs.read` metric could tell you that an `httpd` process has read a total of 320967 bytes from the filesystem, having performed multiple reads over a 10-second period.
* When verbosity is set to higher values, metrics provide details about the action performed by the interposed function in real time. For example, an `fs.read` metric could tell you that an `httpd` process has done one read of 8245 bytes from one specific file, `/etc/httpd/conf/httpd.conf`.
* When the metric name begins with `proc`, metrics periodically report information about resource usage at a point in time. For example, a `proc.mem` metric could tell you that `httpd` is currently using 62,123 KB of memory.

AppScope outputs metrics either in [StatsD](https://github.com/statsd/statsd) format or in equivalent JSON. AppScope can also watch for and intercept StatsD-formatted metrics being emitted by a scoped application.
  
To interpret a given metric, you must consider its type. There are four possibilities:  

| Metric Type | Value Description |
|-------------|-------------------|
| `gauge` | A numeric value that can increase or decrease over time – like a temperature or pressure gauge. |
| `counter` | A cumulative numeric value – it can only increase over time. |
| `timer` | Measures how long a given event type takes (duration). |
| `histogram` | A StatsD histogram distributes sampled observations into buckets. With AppScope, we encounter histograms only in the special case where AppScope intercepts StatsD-formatted metrics whose type is `histogram`. AppScope merely preserves that labeling: we assume that the scoped application has already aggregated the values into buckets. |

For example, `proc.fd` is a `gauge` that indicates how many files were open at one point in time. If we're scoping `top`, that's typically fewer than 10. By contrast, `fs.open` is a `count` that increments every time a file is opened. When scoping `top` over a reporting period of 10 seconds, you could see values in the hundreds or thousands.

It's important to take into account whether the application or command you are scoping is short-lived or longer-lasting. For commands that complete very quickly, a `gauge` will report the value at the moment that AppScope exits the process.

By default, all classes of events and metrics are turned on – but you can turn any class of metric or event data on or off individually. To do this, include or omit the desired watch type(s) from the `metric > watch[*]` array and/or the `event > watch[*]` array in the [config file](/docs/config-file). Environment variables can achieve the same effect. For example, the environment variable to turn off metrics of watch type `statsd` would be `SCOPE_METRIC_STATSD=false`. To turn on events of watch type `logfile`, you'd use `SCOPE_EVENT_LOGFILE=true`.

## The process‑start message

If `configevent` is set to `true`, AppScope sends a process‑start message (the [start.msg](schema-reference#eventstartmsg) event) upon establishing a new connection. This message reports the configs in effect at the time, and also includes four distinct identifiers:
    - **UUID** with key `uuid` and a value in [canonical UUID form](https://en.wikipedia.org/wiki/Universally_unique_identifier). UUID is a universally-unique process identifier, meaning that no two processes will ever have the same UUID value on the same machine.
    - **Machine ID** with key `machine_id` and a value that AppScope obtains from `/etc/machine-id`. The machine ID uniquely identifies the host, as described in the [man page](https://man7.org/linux/man-pages/man5/machine-id.5.html). When `/etc/machine-id` is not available (e.g., in Alpine, or in a container), AppScope generates the machine ID using a repeatable MD5 hash of the host's first MAC address. Two containers on the same host can have the same machine ID.
    - **Process ID**, with key `pid` and a value that is always unique at any given time, but that the machine can reuse for different processes at different times.
    - **ID**, with key `id` and a value that concatenates (and may truncate) the scoped app's hostname, procname, and command. This value is not guaranteed to be unique.

By itself, the **UUID** process ID is unique for a given machine. In principle, **UUID** together with **Machine ID** constitutes a tuple ID that is unique across all machine namespaces.

---
title: Working With AppScope
---

# Working With AppScope

There are three main things to know to work effectively with AppScope:

* Your [overall approach](#ad-hoc-vs-planned) can either be spontaneous or more planned-out.
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

* You are running curl commands against a website, and want to see what changes when you switch between HTTP/1.1 and HTTP/2.2 and/or HTTP and HTTPS. **Use the CLI.** 

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

When AppScope has [interposed a function](how-works), and then the application being scoped executes that function, AppScope can emit events and metrics. Both events and metrics follow patterns which are rigorously defined in validatable [JSON Schema](https://json-schema.org/), and documented in the [Schema Reference](schema-reference).

Events describe the action performed by the interposed function. For example, a `net.open` event could tell you that a network connection was established by Firefox, from the local IP/port `172.16.198.210:54388` to the remote IP/port `34.107.221.82:80`, using HTTP over TCP.

Metrics can do any of three things:
* When verbosity is set to lower values, metrics summarize aspects of the action performed by the interposed function, for a reporting period which defaults to 10 seconds. See the `Metric verbosity level` section of the [config file](config-file).
* When verbosity is set to higher values, metrics provide details about the action performed by the interposed function in real time.
* When the metric name begins with `proc`, metrics provide information about resource usage for the reporting period.

AppScope outputs metrics either in [StatsD](https://github.com/statsd/statsd) format or in equivalent JSON. AppScope can also watch for and intercept StatsD-formatted metrics being emitted by a scoped application.
  
To interpret a given metric, you must consider its type. There are four possibilities:  

| Metric Type | Value Description |
|-------------|-------------------|
| `gauge` | A numeric value that can increase or decrease over time – like a temperature or pressure gauge. |
| `counter` | A cumulative numeric value – it can only increase over time. |
| `timer` | Measures how long a given event type takes (duration). |
| `histogram` | A StatsD histogram distributes sampled observations into buckets. With AppScope, we encounter histograms only in the special case where AppScope intercepts StatsD-formatted metrics whose type is `histogram`. AppScope preserves that labelling without actually aggregating values into buckets. |

For example, `proc.fd` is a `gauge` that indicates how many files were open at one point in time. If we're scoping `top`, that's typically less than 10. By contrast, `fs.open` is a `count` that increments every time a file is opened. When scoping `top` over a reporting period of 10 seconds, you could see values in the hundreds or thousands.

It's important to take into account whether the application or command you are scoping is short-lived or longer-lasting. For commands that complete very quickly, a `gauge` will report the value at the moment that AppScope exits the process.

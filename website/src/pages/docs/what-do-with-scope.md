---
title: Cool, but What Can I do with It?
---

# Cool, but What Can I Do with It?

AppScope offers APM-like, black-box instrumentation of any unmodified Linux executable and application. You can use AppScope in single-user troubleshooting, or in a distributed production deployment, with little extra tooling infrastructure. Especially when paired with [Cribl Stream](https://cribl.io/product/), AppScope can deliver just the data you need to your existing tools.

## Instrument, Collect, and Observe

- Metrics about process and application performance.
- Logs emitted from an application – with zero configuration – delivered to log files or to the console.
- Network flow logs, metrics, and all DNS requests.
- File open and close operations, with I/O consumption per file.
- HTTP requests to and from an application, including URI endpoint, HTTP header, and full payload visibility.
- HTTPS too :)

AppScope works with static or dynamic binaries, and can instrument anything running in Linux. The CLI makes it easy to inspect any application without needing a man-in-the-middle proxy. Once you've gained familiarity, you can use the AppScope library independently of the CLI, with even more fine-grained configuration options.

AppScope collects and forwards StatsD-style metrics about running applications. With HTTP-level visibility, any web server or application can be instantly observable. AppScope's output allows you to use general-purpose tools instead of specialized APM tools and agents.

## Use Cases

You could do any of the following with AppScope (see [Further Examples](examples-use-cases) for details):

- Send HTTP events from Slack to a specified Splunk server.
- Send metrics from nginx to a specified Datadog server.
- Send metrics from a Go static application to a specified Datadog server.
- For any of the above examples, substitute your analytics tool of choice. Optimize the data flow by mediating it through [Cribl Stream](https://cribl.io/product/).
- Run Firefox from the AppScope CLI, and view results on a terminal-based dashboard.
- Run Google Chrome from the AppScope CLI, and view results on a terminal-based dashboard. And be surprised.
- Monitor multiple Apache instances in a server farm, running the data [through Cribl Edge to Cribl Stream](/docs/cribl-integration#scaling-scoped-processes).

People in different roles can solve different problems with AppScope. Consider the following use cases, explored in depth in blog posts by the AppScope team:

- An SRE/developer who needs [application-specific visibility](https://cribl.io/blog/appscope-1-0-changing-the-game-for-sres-and-devs/).
- An Infosec practitioner who needs to [vet an app for security](https://cribl.io/blog/appscope-1-0-changing-the-game-for-infosec-part-1/) before deployment in an enterprise environment.
- An Infosec practitioner who wants to [investigate how apps interact](https://cribl.io/blog/appscope-1-0-changing-the-game-for-infosec-part-2/) in an enterprise environment.

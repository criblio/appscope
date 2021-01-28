# Cool, but What Can I Do with It?

AppScope offers APM-like, black-box instrumentation of any unmodified Linux executable. You can use AppScope in single-user troubleshooting, or in a distributed production deployment, with little extra tooling infrastructure. Especially when paired with Cribl LogStream, AppScope can deliver just the data you need to your existing tools.

Data collection options include:

*   Metrics about performance. 
*   Logs emitted from an application, collected with zero configuration, and delivered to log files or to the console. 
*   Network flow logs and metrics. 
*   DNS Requests. 
*   Files opened and closed, with I/O consumption per file. 
*   HTTP requests to and from an application, including URI endpoint, HTTP header, and full payload visibility.

AppScope works with static or dynamic binaries, to instrument anything running in Linux. The CLI makes it easy to inspect any application without needing a man-in-the-middle proxy. You can use the AppScope library independently of the CLI, applying fine-grained configuration options.

AppScope collects StatsD-style metrics about applications. With HTTP-level visibility, any web server or application can be instantly observable. You get the observability of a proxy/service mesh, without the latency of a sidecar. And you can use general-purpose tools instead of specialized APM tools and agents. 

Some example use cases:

*   Send HTTP events from Slack to a specified Splunk server.
*   Send metrics from nginx to a specified Datadog server.
*   Send metrics from a Go static application to A specified Datadog server.
*   Run Firefox from the AppScope CLI, and view results on a terminal-based dashboard.

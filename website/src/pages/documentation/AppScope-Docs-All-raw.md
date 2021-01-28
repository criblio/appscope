WARNING:
You have 16 H1 headings. You may want to use the "H1 -> H2" option to demote all headings by one level.

----->

<p style="color: red; font-weight: bold">>>>>>  gd2md-html alert:  ERRORs: 0; WARNINGs: 1; ALERTS: 3.</p>
<ul style="color: red; font-weight: bold"><li>See top comment block for details on ERRORs and WARNINGs. <li>In the converted Markdown or HTML, search for inline alerts that start with >>>>>  gd2md-html alert:  for specific instances that need correction.</ul>

<p style="color: red; font-weight: bold">Links to alert messages:</p><a href="#gdcalert1">alert1</a>
<a href="#gdcalert2">alert2</a>
<a href="#gdcalert3">alert3</a>

<p style="color: red; font-weight: bold">>>>>> PLEASE check and correct alert issues and delete this message and the inline alerts.<hr></p>

_Successor to: [https://docs.google.com/document/d/1tzdvtxuKsKGYb4ceShdujB2ThQygIKYn89N15VMZBrI/edit#](https://docs.google.com/document/d/1tzdvtxuKsKGYb4ceShdujB2ThQygIKYn89N15VMZBrI/edit#)_

_The sidebar in docs should follow this sidebar only up to H1 titles. H2 titles and their content are sections within each page._

# § About AppScope

# Overview

## What Is AppScope

AppScope is an open source instrumentation utility for any application, regardless of programming language, with no code modification required.

<p id="gdcalert1" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image1.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert2">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>

![alt_text](images/image1.png "image_tooltip")

![AppScope in-terminal monitoring](../images/AppScope-GUI-screenshot.png)

## Features

AppScope helps users explore, understand, and gain visibility into applications' behavior. Features include:

- Capture application metrics
  - File, Network, Memory, CPU
- Capture and log application events
  - Application console content `stdin/out`
  - Application logs
  - Errors
- Summarize metrics
- Protocol detection
- Capture any and all payloads
  - DNS, HTTP, and HTTPS
- Emit metrics and events to remote systems
- Normalize data
- Monitor static executables
- All runtimes (runtime-agnostic)
- No dependencies
- No code development required

<p id="gdcalert2" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image2.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert3">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>

![alt_text](images/image2.png "image_tooltip")

![AppScope emitting metrics/events to remote systems](../images/AppScope_iso.png)

# Cool, but What Can I Do with It?

AppScope offers APM-like, black-box instrumentation of any unmodified Linux executable. You can use AppScope in single-user troubleshooting, or in a distributed production deployment, with little extra tooling infrastructure. Especially when paired with Cribl LogStream, AppScope can deliver just the data you need to your existing tools.

Data collection options include:

- Metrics about performance.
- Logs emitted from an application, collected with zero configuration, and delivered to log files or to the console.
- Network flow logs and metrics.
- DNS Requests.
- Files opened and closed, with I/O consumption per file.
- HTTP requests to and from an application, including URI endpoint, HTTP header, and full payload visibility.

AppScope works with static or dynamic binaries, to instrument anything running in Linux. The CLI makes it easy to inspect any application without needing a man-in-the-middle proxy. You can use the AppScope library independently of the CLI, applying fine-grained configuration options.

AppScope collects StatsD-style metrics about applications. With HTTP-level visibility, any web server or application can be instantly observable. You get the observability of a proxy/service mesh, without the latency of a sidecar. And you can use general-purpose tools instead of specialized APM tools and agents.

Some example use cases:

- Send HTTP events from Slack to a specified Splunk server.
- Send metrics from nginx to a specified Datadog server.
- Send metrics from a Go static application to A specified Datadog server.
- Run Firefox from the AppScope CLI, and view results on a terminal-based dashboard.

# How AppScope Works

AppScope consists of a shared library and an executable. In simple terms, you load the library into the address space of a process. With the library loaded, details are extracted from applications as they execute. The executable provides a command line interface (CLI), which can optionally be used to control which processes are interposed, what data is extracted, and how to view and interact with the results.

The library extracts information as it interposes functions. When an application calls a function, it actually calls a function of the same name in the AppScope library. AppScope extracts details from the function, and the original function call proceeds. This interposition of function calls requires no change to an application: it works with unmodified binaries, and its CPU and memory overhead are minimal.

The library is loaded using a number of mechanisms, depending on the type of executable. A dynamic loader can preload the library (where supported) and AppScope is used to load static executables.

Child processes are created with the library present, if the library was present in the parent. In this manner, a single executable is able to start, daemonize, and create any number of children, all of which include interposed functions.

<p id="gdcalert3" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image3.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert4">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>

![alt_text](images/image3.png "image_tooltip")

![AppScope system-level design](../images/AppScope-system-level-design.png)

# AppScope Components

AppScope consists of three components:

1. Command Line Interface \
   The AppScope CLI (`scope`) provides a quick and easy way to explore capabilities and to obtain insight into application behavior. Here, no installation or configuration is required to get started exploring application behavior.
2. Loader \
   Linux, like other operating systems, provides a loader capable of loading and linking dynamic and static executables as needed. AppScope provides a very simple component (`ldscope`) that supports loading static executables. This allows AppScope to interpos functions, and to thereby expose the same data that is exposed from dynamic executables. (While the AppScope loader is optional for use with dynamic executables, it is required in order when used to extract details from a static executable.) \

3. Library \
   The AppScope library (`libscope`) is the core component that resides in application processes, extracting data as an application executes. You can closely configure the library's behavior using environment variables or a configuration file.

# Use Cases by Component

## Get Started with the CLI

The easiest way to get started with AppScope is to use the CLI. This provides a rich set of capabilities intended to capture data from single applications. Data is captured in the local file system and is managed through the CLI.

## Use the Loader Interface (Production Environments)

As you get more interested in obtaining details from production applications, explore using the AppScope library apart from the CLI. This allows for full configurability via the `scope.yml` configuration file (default `scope.yml` [here](<[https://github.com/criblio/appscope/blob/master/conf/scope.yml](https://github.com/criblio/appscope/blob/master/conf/scope.yml)l>)). You can send details over UDP, TCP, and local or remote connections, and you can define specific events and metrics for export. You can use the AppScope loader to start applications that include the library.

## Use the Library Independently (Production Environments)

You can also load the AppScope library independently, configured by means of a configuration file and/or environment variables. The config file (default: [`scope.yml`](https://github.com/criblio/appscope/blob/master/conf/scope.yml)) can be located where needed – just reference its location using the `LD_PRELOAD` environment variable. Environment variables take precedence over the default configuration, as well as over details in any configuration file.

You use an environment variable to load the library independent of any executable. Whether you use this option or the AppScope loader, you get full control of the data source, formats, and transports.

# § Installing AppScope

# Requirements

Requirements for AppScope are as follows:

## Operating Systems (Linux only)

Supported:

- RedHat Enterprise Linux or CentOS 6.4 and later
- Ubuntu 16 and later
- Amazon Linux 1 and 2
- Debian

CPU: x84-64 architecture

Memory: 1GB

Disk: 20MB (library + CLI)

## Known Limitations

These runtimes are not supported Open JVM &lt; v.6, Oracle JVM &lt; v.6, Go &lt; v.1.8.

# Quick Start Guide

First, see &lt;link>[Requirements](#bookmark=id.2gpv5bl9l16m)&lt;/link> to ensure that you’re completing these steps on a supported system. Getting started is easy.

## Get AppScope

Directly download the CLI binary from [https://cdn.cribl.io/dl/scope/cli/linux/scope](https://s3-us-west-2.amazonaws.com/io.cribl.cdn/dl/scope/cli/linux/scope). Use this curl command:

```

curl -Lo scope https://cdn.cribl.io/dl/scope/cli/linux/scope && chmod 755 ./scope

```

## Explore the CLI

Run `scope --help` or `scope -h` to view CLI options. Also see the complete &lt;link>[CLI Reference](#bookmark=id.q6rt37xg7u0g)&lt;/link>.

## Scope Some Commands

Test-run one or more well-known Linux commands, and view the results. E.g.:

```

scope curl https://google.com

scope ps -ef

```

## Explore the Data

To see the monitoring and visualization features AppScope offers, exercise some of its options. E.g.:

Show captured metrics:

`scope metrics`

Plot a chart of the `proc.cpu` metric:

`scope metrics -m proc.cpu -g`

Display captured events:

`scope events`

Filter out events, for just http:

`scope events -t http`

List this AppScope session's history:

`scope history`

## Next Steps

For guidance on taking AppScope to the next level, [j[oin](https://cribl.io/community/)](https://cribl.io/community/) our [[community on Slack](https://app.slack.com/client/TD0HGJPT5/CPYBPK65V/thread/C01BM8PU30V-1611001888.001100)](https://app.slack.com/client/TD0HGJPT5/CPYBPK65V/thread/C01BM8PU30V-1611001888.001100).

# Updating

You can update AppScope in place. First, download a fresh version of the binary:

```

cd &lt;your-appscope-directory>

curl -Lo scope https://s3-us-west-2.amazonaws.com/io.cribl.cdn/dl/scope/cli/linux/scope && chmod 755 ./scope

```

Next, confirm the overwrite.

After updating, verify AppScope's version and build date:

`scope version`

# Uninstalling

You can uninstall AppScope by simply deleting the code:

`rm -rf appscope`

… and the associated history directory:

``

`cd ~

rm -rf ./scope

```

Currently scope’d applications will continue to run. To remove the AppScope library from a running process, you will need to restart that process.


# § Using AppScope


# Using the CLI

You can use the AppScope CLI to monitor applications and other processes, as shown in these examples:

Generate a large data set:

`scope firefox`

Scope every subsequent shell command:

`scope bash`

If you have [Cribl LogStream](https://cribl.io/download/) installed, try:

`scope /&lt;cribl_home>/bin/cribl server`

To release a process, like `scope bash`, `exit` the process in the shell. If you do this with the bash shell itself, you won't be able to `scope bash` again in the same terminal session.


# CLI Reference


## ## Command Syntax

To execute CLI commands, the basic syntax is:

```

./scope &lt;command> [flags] [options]

```


## ## Commands Available

To see a list of available commands, enter `./scope` alone, or `./scope -h`, or  `./scope --help`.  This displays the basic help listing below.

```

Command line interface for working with Cribl AppScope

Usage:

scope [command]

Available Commands:

dash Display scope dashboard

events Output events for a session

help Help about any command

history List scope session history

metrics Output metrics for a session

prune Delete scope history

run Execute a scoped command

version Display scope version

Flags:

-h, --help Help for scope

-v, --verbose count Set verbosity level

Use "scope [command] --help" for more information about a command.

```

As noted just above, to see a specific command's help or its required parameters, enter `./scope &lt;command> -h` or `./scope help &lt;command> [flags]`.


### ### `dash`

Displays an interactive dashboard with an overview of what's happening with the selected session.


#### #### Usage

`scope dash [flags]`


#### #### Examples

`scope dash`


#### #### Flags

```

-h, --help help for dash

-i, --id int Display info from specific from session ID (default -1)

```


### ### `events`

Outputs events for a session. You can obtain detailed information about each event by inputting the Event ID as a positional parameter. (By default, the Event ID appears in blue in []'s at the left.) You can provide filters to narrow down by name (e.g., `http`, `net`, `fs`, or `console`) or by field (e.g., `fs.open`, `stdout`, or `net.conn.open`). You can use JavaScript expressions to further refine the query and express logic.


#### #### Usage

`scope events [flags] ([eventId])`


#### #### Examples

```

scope events

scope events -t http

scope events -s stderr

scope events -e 'sourcetype!="net"'

scope events -n 1000 -e 'sourcetype!="console" && source.indexOf("cribl.log") == -1 && (data["file.name"] || "").indexOf("/proc") == -1'

```


#### #### Flags

```

-a, --all Show all events

      --allfields            Displaying hidden fields

      --color                Force color on (if tty detection fails or pipeing)

-e, --eval string Evaluate JavaScript expression against event. Must return truthy to print event.

-f, --follow Follow a file, like tail -f

-h, --help help for events

-i, --id int Display info from specific from session ID (default -1)

-j, --json Output as newline delimited JSON

-n, --last int Show last &lt;n> events (default 20)

-m, --match string Display events containing supplied string

-s, --source strings Display events matching supplied sources

-t, --sourcetype strings Display events matching supplied sourcetypes

```


### ### `history`

Lists scope session history.


#### #### Usage

`scope history [flags]`


#### #### Aliases

`history, hist`


#### #### Examples

`scope history`


#### #### Flags

```

-a, --all List all sessions

-d, --dir Output just directory (with -i)

-h, --help help for history

-i, --id int Display info from specific from session ID (default -1)

-n, --last int Show last &lt;n> sessions (default 20)

-r, --running List running sessions

```


### ### `metrics`

Outputs metrics for a session.


#### #### Usage

`scope metrics [flags]`


#### #### Examples

`scope metrics`


#### #### Flags

```

-c, --cols Display metrics as columns

-g, --graph Graph this metric

-h, --help help for metrics

-i, --id int Display info from specific from session ID (default -1)

-m, --metric strings Display only supplied metrics

-u, --uniq Display first instance of each unique metric

```


### ### `prune`

Deletes scope history for this session.


#### #### Usage

`scope prune [flags]`


#### #### Examples

```

scope prune -k 20

scope prune -a

```


#### #### Flags

```

-a, --all Delete all sessions

-d, --delete int Delete last &lt;delete> sessions (default -1)

-f, --force Do not prompt for confirmation

-h, --help help for prune

-k, --keep int Keep last &lt;keep> sessions (default -1)

```


### ### `run`

Executes a scoped command (or application.


#### #### Usage

`scope run [flags]`


#### #### Examples

`scope run /bin/echo "foo"`


#### #### Flags

```

-h, --help help for run

      --passthrough     Runs scopec with current environment & no config.

-p, --payloads Capture payloads of network transactions

-v, --verbosity int Set scope metric verbosity (default 4)

```


### ### `version`

Outputs version info.


#### #### Usage

`scope version [flags]`


#### #### Examples

```

scope version

scope version --date

scope version --summary

```


#### #### Flags

```

      --date      output just the date

-h, --help help for version

      --summary   output just the summary

```


### ### `--verbose`

This flag sets the verbosity level. `0` is least verbose, `4` is the default, and `9` is most default. For descriptions of individual levels, see  &lt;link>[Appendix: Default Configuration](https://docs.google.com/document/d/1tzdvtxuKsKGYb4ceShdujB2ThQygIKYn89N15VMZBrI/edit#bookmark=id.xnkgdpnlvncq)&lt;/link>.


#### #### Usage

```

scope --verbose &lt;level>

scope -v &lt;level>

```


#### #### Examples

```

scope --verbose &lt;level>

scope -v &lt;level>

```


# Using the Library

To use the AppScope library independently of the CLI or loader, you rely on the `LD_PRELOAD` environment variable. This section provides several examples – all calling the system-level `ps` command – simply to show how the syntax works.


## LD_PRELOAD Environment Variable with a Single Command

Start with this basic example:

`LD_PRELOAD=./libscope.so ps -ef`

This executes the command `ps -ef`. But first, the OS' loader loads the AppScope library as part of loading and linking the ps executable.

Details of the ps application's execution are emitted to the configured transport, in the configured format. For configuration details, see [Configuring the Library](#configuring).


## LD_PRELOAD Environment Variable Extended Examples

These examples demonstrate using `LD_PRELOAD` with additional variables.



1. `LD_PRELOAD=./libscope.so SCOPE_METRIC_VERBOSITY=5 ps -ef`

This again executes the `ps` command using the AppScope library. But it also defines the verbosity for metric extraction as level 5. (This verbosity setting overrides any config-file setting, as well as the default value.)



2. `LD_PRELOAD=./libscope.so SCOPE_HOME=/etc/scope ps -ef`

This again executes the `ps` command using the AppScope library. But it also directs the library to use the config file `/etc/scope/scope.yml`.



3. `LD_PRELOAD=./libscope.so SCOPE_EVENT_DEST=tcp://localhost:9999 ps -ef`

This again executes the `ps` command using the AppScope library. But here, we also specify that events (as opposed to metrics) will be sent over a TCP connection to localhost, using port 9999. (This event destination setting overrides any config-file setting, as well as the default value.)


# &lt;span id="configuring"> Configuring the Library &lt;/span>

For details on configuring the library, see AppScope online help's CONFIGURATION section. For the default settings in the sample `scope.yml` configuration file, see the &lt;link>Appendix&lt;/link>, or see the most-recent file on [GitHub]([https://github.com/criblio/appscope/blob/master/conf/scope.yml](https://github.com/criblio/appscope/blob/master/conf/scope.yml)).


# Examples and Use Cases

Here are some examples of using AppScope to monitor specific applications, and to send the resulting data to specific destinations.



1. Run Firefox from the AppScope CLI, and view results on a terminal-based dashboard:

```

scope firefox &

scope dash

```



2. Send metrics from nginx to Datadog at `ddoghost`, using an environment variable and all other defaults:

```

SCOPE_METRIC_DEST=udp://ddoghost:5000 ldscope nginx

```



3. Send configured events from curl to the server `mydata` on port 9000, using the config file at `/opt/scope/scope.yml`:

```

SCOPE_HOME=/opt/scope ldscope curl https://cribl.io

```

Configuration file example:

```

event:

enable: true

transport:

    type: tcp

    host: mydata

    port: 9000

```



4. Send HTTP events from Slack to a Splunk server at `shost`:

```

ldscope slack

```

Configuration file example, located at `/etc/scope` or `~/scope.yml`:

```

event:

enable: true

transport:

    type: tcp

    host: shost

    port: 8088

---

    - type: http

      name: .*

      field: .*

      value: .*

```



5. Send DNS events from curl to the default host. This example uses the library in the current directory, independently of the CLI or loader:

```

SCOPE_EVENT_DNS=true LD_PRELOAD=./libscope.so curl [https://cribl.io](https://cribl.io)

```



6. Send default metrics from the Go static application `hello` to the Datadog server at `ddog`:

```

ldscope ./hello

```

Configuration file example:

```

metric:

enable: true

format:

    type : statsd

---

transport:

    type: udp

    host: ddog

    port: 5000

```


# § Community


# Join Our Slack

To connect with other AppScope users and developers, [[join](https://cribl.io/community/)](https://cribl.io/community/) our [[Community Slack](https://app.slack.com/client/TD0HGJPT5/CPYBPK65V/thread/C01BM8PU30V-1611001888.001100)](https://app.slack.com/client/TD0HGJPT5/CPYBPK65V/thread/C01BM8PU30V-1611001888.001100).


# How to Contribute

Fork our [[repo on GitHub](https://github.com/criblio/appscope.git)](https://github.com/criblio/appscope.git).


# § Appendix: Default Configuration

Below are the contents of the default AppScope library configuration file. (You can access this default file's most-recent version [on GitHub](https://github.com/criblio/appscope/blob/master/conf/scope.yml).)

---

metric:

  enable: true                      # true, false

  format:

    type : statsd                   # statsd, ndjson

    #statsdprefix : 'cribl.scope'    # prepends each StatsD metric

    statsdmaxlen : 512              # max size of a formatted StatsD string

    verbosity : 4                   # 0-9 (0 is least verbose, 9 is most)

          # 0-9 controls which expanded tags are output

          #      1 "data"

          #      1 "unit"

          #      2 "class"

          #      2 "proto"

          #      3 "op"

          #      4 "host"

          #      4 "proc"

          #      5 "domain"

          #      5 "file"

          #      6 "localip"

          #      6 "remoteip"

          #      6 "localp"

          #      6 "port"

          #      6 "remotep"

          #      7 "fd"

          #      7 "pid"

          #      8 "duration"

          #      9 "numops"

          #

          # 5-9 disables event summarization of different kinds

          #      5  turns off error event summarization

          #      6  turns off filesystem open/close event summarization

          #      6  turns off dns event summarization

          #      7  turns off filesystem stat event summarization

          #      7  turns off network connect event summarization

          #      8  turns off filesystem seek event summarization

          #      9  turns off filesystem read/write summarization

          #      9  turns off network send/receive summarization

    tags:

      #user: $USER

      #feeling: elation

  transport:                        # defines how scope output is sent

    type: udp                       # udp, tcp, unix, file, syslog

    host: 127.0.0.1

    port: 8125

event:

  enable: true                      # true, false

  transport:

    type: tcp                       # udp, tcp, unix, file, syslog

    host: 127.0.0.1

    port: 9109

  format:

    type : ndjson                   # ndjson

    maxeventpersec: 10000           # max events per second.  zero is "no limit"

    enhancefs: true                 # true, false

  watch:

    # Creates events from data written to files.

    # Designed for monitoring log files, but capable of capturing

    # any file writes.

#    - type: file

#      name: .*log.*                 # whitelist ex regex describing log file names

#      value: .*                     # whitelist ex regex describing field values

    # Creates events from data written to stdout, stderr, or both.

    # May be most useful for capturing debug output from processes

    # running in containerized environments.

#    - type: console

#      name: stdout                  # (stdout|stderr)

#      value: .*

    # Creates events from libscope's metric (statsd) data.  May be

    # most useful for data which could overwhelm metric aggregation

    # servers, either due to data volumes or dimensions (tags) with

    # high cardinality.

#    - type: metric

#      name: .*                      # (net.*)|(.*err.*)

#      field: .*                     # whitelist regex describing field names

#      value: .*

    # Enable extraction of HTTP headers

#    - type: http

#      name: .*                      # (http-resp)|(http-metrics)

#      field: .*                     # whitelist regex describing field names

#      value: .*

    # Creates events describing network connectivity

#    - type: net

#      name: .*                      #

#      field: .*                     # whitelist regex describing field names

#      value: .*

    # Creates events describing file connectivity

#    - type: fs

#      name: .*                      #

#      field: .*                     # whitelist regex describing field names

#      value: .*

    # Creates events describing dns activity

#    - type: dns

#      name: .*                      #

#      field: .*                     # whitelist regex describing field names

#      value: .*

payload:

  enable: false                     # true, false

  dir: '/tmp'

libscope:

  configevent: true                 # true, false

  summaryperiod : 10                # in seconds

  commanddir : '/tmp'

  #  commanddir supports changes to configuration settings of running

  #  processees.  At every summary period the library looks in commanddir

  #  to see if a file named scope.&lt;pid> exists.  (where pid is the process ID

  #  of the process running with scope.)  If it exists, it processes every

  #  line for environment variable-style commands:

  #      SCOPE_METRIC_VERBOSITY=9

  #  It changes the configuration to match the new settings, and deletes the

  #  scope.&lt;pid> file when it's complete.

  log:

    level: error                    # debug, info, warning, error, none

    transport:

      type: file

      path: '/tmp/scope.log'

      buffering: line               # line, full

...


# § AppScope Help Text

To export all the text below, download [http://cdn.cribl.io/dl/scope/latest/linux/libscope.so](http://cdn.cribl.io/dl/scope/latest/linux/libscope.so) onto a Linux system, and execute: `./libscope.so all > AppScope-help.all.txt`

Related: [AppSoope README.md remix](https://docs.google.com/document/d/1e5OkQ_bEwCFvX4-crTYxgxz4ig5YlkblI_IFZOFltTU/edit?ts=6006211c).



---


    OVERVIEW:

    The Scope library supports extraction of data from within applications.

    As a general rule, applications consist of one or more processes.

    The Scope library can be loaded into any process as the

    process starts.

    The primary way to define which processes include the Scope library

    is by exporting the environment variable LD_PRELOAD, which is set to point to the path name of the Scope library. E.g.:

    export LD_PRELOAD=./libscope.so

    Scope emits data as metrics and/or events.

    Scope is fully configurable by means of a configuration file (scope.yml) and/or

    environment variables.

    Metrics are emitted in StatsD format, over a configurable link. By default,

    metrics are sent over a UDP socket using localhost and port 8125.

    Events are emitted in JSON format over a configurable link. By default,

    events are sent over a TCP socket using localhost and port 9109.

    Scope logs to a configurable destination, at a configurable

    verbosity level. The default verbosity setting is level 4, and the

    default destination is the file `/tmp/scope.log`.

    CONFIGURATION:

    Configuration File:

       A YAML config file (named scope.yml) enables control of all available settings.

       The config file is optional. Environment variables take precedence

       over settings in a config file.

    Config File Resolution

        If the SCOPE_CONF_PATH env variable is defined, and points to a

        file that can be opened, it will use this as the config file. Otherwise, AppScope searches for the config file in this priority order, using

        the first one it finds:

            $SCOPE_HOME/conf/scope.yml

            $SCOPE_HOME/scope.yml

            /etc/scope/scope.yml

            ~/conf/scope.yml

            ~/scope.yml

            ./conf/scope.yml

            ./scope.yml



    Environment Variables:

    SCOPE_CONF_PATH

        Directly specify config file's location and name.

        Used only at start time.

    SCOPE_HOME

        Specify a directory from which conf/scope.yml or ./scope.yml can

        be found. Used only at start time, and only if SCOPE_CONF_PATH does

        not exist. For more info, see Config File Resolution below.

    SCOPE_METRIC_ENABLE

        Single flag to make it possible to disable all metric output.

        true,false  Default is true.

    SCOPE_METRIC_VERBOSITY

        0-9 are valid values. Default is 4.

        For more info, see Metric Verbosity below.

    SCOPE_METRIC_DEST

        Default is udp://localhost:8125

        Format is one of:

            file:///tmp/output.log   (file://stdout, file://stderr are

                                      special allowed values)

            udp://&lt;server>:&lt;123>         (&lt;server> is servername or address;

                                      &lt;123> is port number or service name)

    SCOPE_METRIC_FORMAT

        statsd, ndjson

        Default is statsd.

    SCOPE_STATSD_PREFIX

        Specify a string to be prepended to every scope metric.

    SCOPE_STATSD_MAXLEN

        Default is 512.

    SCOPE_SUMMARY_PERIOD

        Number of seconds between output summarizations. Default is 10.

    SCOPE_EVENT_ENABLE

        Single flag to make it possible to disable all event output.

        true,false  Default is true.

    SCOPE_EVENT_DEST

        Same format as SCOPE_METRIC_DEST above.

        Default is tcp://localhost:9109

    SCOPE_EVENT_FORMAT

        ndjson

        Default is ndjson.

    SCOPE_EVENT_LOGFILE

        Create events from writes to log files.

        true,false  Default is false.

    SCOPE_EVENT_LOGFILE_NAME

        An extended regex to filter log file events by file name.

        Used only if SCOPE_EVENT_LOGFILE is true. Default is .*log.*

    SCOPE_EVENT_LOGFILE_VALUE

        An extended regex to filter log file events by field value.

        Used only if SCOPE_EVENT_LOGFILE is true. Default is .*

    SCOPE_EVENT_CONSOLE

        Create events from writes to stdout, stderr.

        true,false  Default is false.

    SCOPE_EVENT_CONSOLE_NAME

        An extended regex to filter console events by event name.

        Used only if SCOPE_EVENT_CONSOLE is true. Default is .*

    SCOPE_EVENT_CONSOLE_VALUE

        An extended regex to filter console events by field value.

        Used only if SCOPE_EVENT_CONSOLE is true. Default is .*

    SCOPE_EVENT_METRIC

        Create events from metrics.

        true,false  Default is false.

    SCOPE_EVENT_METRIC_NAME

        An extended regex to filter metric events by event name.

        Used only if SCOPE_EVENT_METRIC is true. Default is .*

    SCOPE_EVENT_METRIC_FIELD

        An extended regex to filter metric events by field name.

        Used only if SCOPE_EVENT_METRIC is true. Default is .*

    SCOPE_EVENT_METRIC_VALUE

        An extended regex to filter metric events by field value.

        Used only if SCOPE_EVENT_METRIC is true. Default is .*

    SCOPE_EVENT_HTTP

        Create events from HTTP headers.

        true,false  Default is false.

    SCOPE_EVENT_HTTP_NAME

        An extended regex to filter http events by event name.

        Used only if SCOPE_EVENT_HTTP is true. Default is .*

    SCOPE_EVENT_HTTP_FIELD

        An extended regex to filter http events by field name.

        Used only if SCOPE_EVENT_HTTP is true. Default is .*

    SCOPE_EVENT_HTTP_VALUE

        An extended regex to filter http events by field value.

        Used only if SCOPE_EVENT_HTTP is true. Default is .*

    SCOPE_EVENT_NET

        Create events describing network connectivity.

        true,false  Default is false.

    SCOPE_EVENT_NET_NAME

        An extended regex to filter network events by event name.

        Used only if SCOPE_EVENT_NET is true. Default is .*

    SCOPE_EVENT_NET_FIELD

        An extended regex to filter network events by field name.

        Used only if SCOPE_EVENT_NET is true. Default is .*

    SCOPE_EVENT_NET_VALUE

        An extended regex to filter network events by field value.

        Used only if SCOPE_EVENT_NET is true. Default is .*

    SCOPE_EVENT_FS

        Create events describing file connectivity.

        true,false  Default is false.

    SCOPE_EVENT_FS_NAME

        An extended regex to filter file events by event name.

        Used only if SCOPE_EVENT_FS is true. Default is .*

    SCOPE_EVENT_FS_FIELD

        An extended regex to filter file events by field name.

        Used only if SCOPE_EVENT_FS is true. Default is .*

    SCOPE_EVENT_FS_VALUE

        An extended regex to filter file events by field value.

        Used only if SCOPE_EVENT_FS is true. Default is .*

    SCOPE_EVENT_DNS

        Create events describing DNS activity.

        true,false  Default is false.

    SCOPE_EVENT_DNS_NAME

        An extended regex to filter dns events by event name.

        Used only if SCOPE_EVENT_DNS is true. Default is .*

    SCOPE_EVENT_DNS_FIELD

        An extended regex to filter DNS events by field name.

        Used only if SCOPE_EVENT_DNS is true. Default is .*

    SCOPE_EVENT_DNS_VALUE

        An extended regex to filter dns events by field value.

        Used only if SCOPE_EVENT_DNS is true. Default is .*

    SCOPE_EVENT_MAXEPS

        Limits number of events that can be sent in a single second.

        0 is "no limit"; 10000 is the default.

    SCOPE_ENHANCE_FS

        Controls whether uid, gid, and mode are captured for each open.

        Used only if SCOPE_EVENT_FS is true. true,false Default is true.

    SCOPE_LOG_LEVEL

        debug, info, warning, error, none. Default is error.

    SCOPE_LOG_DEST

        same format as SCOPE_METRIC_DEST above.

        Default is file:///tmp/scope.log

    SCOPE_TAG_

        Specify a tag to be applied to every metric. Environment variable

        expansion is available, e.g.: SCOPE_TAG_user=$USER

    SCOPE_CMD_DIR

        Specifies a directory to look for dynamic configuration files.

        See Dynamic Configuration below.

        Default is /tmp

    SCOPE_PAYLOAD_ENABLE

        Flag that enables payload capture.  true,false  Default is false.

    SCOPE_PAYLOAD_DIR

        Specifies a directory where payload capture files can be written.

        Default is /tmp

    SCOPE_CONFIG_EVENT

        Sends a single process-identifying event, when a transport

        connection is established.  true,false  Default is true.

    Dynamic Configuration:

        Dynamic Configuration allows configuration settings to be

        changed on the fly after process start time. At every

        SCOPE_SUMMARY_PERIOD, the library looks in SCOPE_CMD_DIR to

        see if a file scope.&lt;pid> exists. If it exists, the library processes

        every line, looking for environment variable–style commands

        (e.g., SCOPE_CMD_DBG_PATH=/tmp/outfile.txt). The library changes the

        configuration to match the new settings, and deletes the

        scope.&lt;pid> file when it's complete.

    METRICS:

    Metrics can be enabled or disabled with a single config element

    (metric: enable: true|false). Specific types of metrics, and specific

    field content, are managed with a Metric Verbosity setting.

    Metric Verbosity

        Controls two different aspects of metric output –

        Tag Cardinality and Summarization.

        Tag Cardinality

            0   No expanded StatsD tags

            1   adds 'data', 'unit'

            2   adds 'class', 'proto'

            3   adds 'op'

            4   adds 'pid', 'host', 'proc', 'http_status'

            5   adds 'domain', 'file'

            6   adds 'localip', 'remoteip', 'localp', 'port', 'remotep'

            7   adds 'fd', 'args'

            8   adds 'duration', 'numops', 'req_per_sec', 'req', 'resp', 'protocol'

        Summarization

            0-4 has full event summarization

            5   turns off 'error'

            6   turns off 'filesystem open/close' and 'dns'

            7   turns off 'filesystem stat' and 'network connect'

            8   turns off 'filesystem seek'

            9   turns off 'filesystem read/write' and 'network send/receive'

    The http.status metric is emitted when the http watch type has been

    enabled as an event. The http.status metric is not controlled with

    summarization settings.

    EVENTS:

    All events can be enabled or disabled with a single config element \
    (event: enable: true|false). Unlike metrics, event content is not

    managed with verbosity settings. Instead, you use regex filters that

    manage which field types and field values to include.

     Events are organized as 7 watch types:

     1) File Content. Provide a pathname, and all data written to the file

        will be organized in JSON format and emitted over the event channel.

     2) Console Output. Select stdin and/or stdout, and all data written to

        these endpoints will be formatted in JSON and emitted over the event

        channel.

     3) Metrics. Event metrics provide the greatest level of detail from libscope. Events are created for every read, write, send, receive, open, close, and connect. These raw events are configured with regex filters to manage which event, which specific fields within an event, and which value patterns within a field to include.

     4) HTTP Headers. HTTP headers are extracted, formatted in JSON, and

        emitted over the event channel. Three types of events are created

        for HTTP headers: 1) HTTP request events, 2) HTTP response events,

        and 3) a metric event corresponding to the request and response

        sequence. A response event includes the corresponding request,

        status and duration fields. An HTTP metric event provides fields

        describing bytes received, requests per second, duration, and status.

     5) File System. Events are formatted in JSON for each file system open, including file name, permissions, and cgroup. Events for file system close add a summary of the number of bytes read and written, the total number of read and write operations, and the total duration of read and write operations. The specific function performing open and close is reported as well.

     6) Network. Events are formatted in JSON for network connections and corresponding disconnects, including type of protocol used, and local and peer IP:port. Events for network disconnect add a summary of the number of bytes sent and received, and the duration of the sends and receives while the connection was active. The reason (source) for disconnect is provided as local or remote.

     7) DNS.   Events are formatted in JSON for DNS requests and responses. The event provides the domain name being resolved. On DNS response, the event provides the duration of the DNS operation.

    PROTOCOL DETECTION:

     Scope can detect any defined network protocol. You provide protocol

     definitions in a separate YAML config file (which should be named scope_protocol.yml). You describe protocol specifics

     in one or more regex definitions. PCRE2 regular

     expressions are supported. You can find a sample config file at https://github.com/criblio/appscope/blob/master/conf/scope_protocol.yml.

     Scope detects binary and string protocols. Detection events,

     formatted in JSON, are emitted over the event channel. Enable the

      event metric watch type to allow protocol detection.

     The protocol detection config file should be named scope_protocol.yml.

     Place the protocol definitions config file (scope_protocol.yml) in the directory defined by the SCOPE_HOME

     environment variable. If Scope does not find the protocol definitions

     file in that directory, it will search for it, in the same search order as described for config files.

PAYLOAD EXTRACTION:

When enabled, libscope extracts payload data from network operations. Payloads are emitted in binary. No formatting is applied to the data. Payloads are emitted to either a local file or the LogStream channel. Configuration elements for libscope support defining a path for payload data.
```

---
title: CLI Reference
---

## CLI Reference
---

In the AppScope CLI, the `scope` command takes a Linux command as an argument. That's called "scoping" the Linux command. For example, if you run `scope top`, we say you have "scoped" the `top` command.

The AppScope CLI also has subcommands, which pair with `scope` to do many things. For example, if you run `scope dash`, AppScope displays its dashboard.

This Reference explains how to use `scope` and its arsenal of subcommands.

### Subcommand Syntax

To execute CLI subcommands, the basic syntax is:

```
./scope <subcommand> [flags] [options]
```

### Subcommands Available

To see a list of available subcommands, enter `./scope` alone, or `./scope -h`, or `./scope --help`. This displays the basic help listing below.

```
Cribl AppScope Command Line Interface

AppScope is a general-purpose observable application telemetry system.

Running `scope` with no subcommands will execute the `scope run` command.

Usage:
  scope [subcommand]

Available Subcommands:
  attach      Scope a currently-running process
  completion  Generate completion code for specified shell
  dash        Display scope dashboard
  detach      Unscope a currently-running process
  events      Outputs events for a session
  extract     Output instrumentary library files to <dir>
  flows       Observed flows from the session, potentially including payloads
  help        Help about any subcommand
  history     List scope session history
  k8s         Install scope in kubernetes
  logs        Display scope logs
  metrics     Outputs metrics for a session
  prune       Prune deletes session history
  ps          List processes currently being scoped
  run         Executes a scoped command
  service     Configure a systemd/OpenRC service to be scoped
  start       Start scoping a filtered selection of processes and services
  version     Display scope version
  watch       Executes a scoped command on an interval

Flags:
  -h, --help   Help for scope

Use "scope [subcommand] --help" for more information about a subcommand.
```

As noted just above, to see a specific subcommand's help or its required parameters, enter: 
`./scope <subcommand> -h` 

…or: 
`./scope help <subcommand> [flags]`.

---

### attach
---

Scopes a currently-running process identified by PID or ProcessName.

#### Usage

`scope attach [flags] PID | <process_name>`

#### Examples

```
scope attach 1000
scope attach firefox
scope attach --payloads 2000
```

#### Flags

```
  -a, --authtoken string      Set AuthToken for Cribl Stream
  -c, --cribldest string      Set Cribl Stream destination for metrics & events (host:port defaults to tls://)
  -e, --eventdest string      Set destination for events (host:port defaults to tls://)
  -h, --help                  Help for attach
  -l, --librarypath string    Set path for dynamic libraries
      --loglevel string       Set ldscope log level (debug, warning, info, error, none)
  -m, --metricdest string     Set destination for metrics (host:port defaults to tls://)
      --metricformat string   Set format of metrics output (statsd|ndjson); default is "ndjson"
  -n, --nobreaker             Set Cribl Stream to not break streams into events
      --passthrough           Runs ldscope with current environment & no config
  -p, --payloads              Capture payloads of network transactions
  -u, --userconfig string     Run ldscope with a user specified config file; overrides all other settings
  -v, --verbosity int         Set scope metric verbosity (default 4)

```

### detach
---

Unscopes a currently-running process identified by PID or ProcessName.

#### Usage

`scope detach [flags] PID | <process_name>`

#### Examples

```
scope detach 1000
scope detach firefox
```

#### Flags

```
  -h, --help                  Help for detach

```

### completion
---

Generates completion code for specified shell.

#### Usage

`scope completion [flags] [bash|zsh]`

#### Examples

```
scope completion bash > /etc/bash_completion.d/scope  # Generates and installs scope autocompletion for bash
source <(scope completion bash)                       # Generates and loads scope autocompletion for bash
```

#### Flags

```
  -h, --help   Help for completion
  
```


### dash
---

Displays an interactive dashboard with an overview of what's happening with the selected session.

#### Usage

`scope dash [flags]`

#### Examples

`scope dash`

#### Flags

```
  -h, --help     Help for dash
  -i, --id int   Display info from specific from session ID (default -1)
```

### events
---
Outputs events for a session. You can obtain detailed information about each event by inputting the Event ID as a positional parameter. (By default, the Event ID appears in blue, in `[]`'s at the left.) You can provide filters to narrow down by name (e.g., `http`, `net`, `fs`, or `console`), or by field (e.g., `fs.open`, `stdout`, or `net.open`). You can use JavaScript expressions to further refine the query, and to express logic.

#### Usage

`scope events [flags] ([eventId])`

#### Examples

```
Examples:
scope events
scope events m61
scope events --sourcetype http
scope events --source stderr
scope events --match file
scope events --fields net_bytes_sent,net_bytes_recv --match net_bytes
scope events --follow
scope events --all
scope events --allfields
scope events --id 4
scope events --sort _time --reverse
scope events --eval 'sourcetype!="net"'
scope events -n 1000 -e 'sourcetype!="console" && source.indexOf("cribl.log") == -1 && (data["file.name"] || "").indexOf("/proc") == -1'
```

#### Flags

```
  -a, --all                  Show all events
      --allfields            Displaying hidden fields
      --color                Force color on (if tty detection fails or piping)
  -e, --eval string          Evaluate JavaScript expression against event. Must return truthy to print event.
                             Note: Post-processes after matching, not guaranteed to return last <n> events.
      --fields strings       Display the names and values for specified fields only, for each event (look at JSON output for field names)
  -f, --follow               Follow a file, like tail -f
  -h, --help                 help for events
  -i, --id int               Display info from specific from session ID (default -1)
  -j, --json                 Output as newline delimited JSON
  -n, --last int             Show last <n> events (default 20)
  -m, --match string         Display events containing supplied string
  -r, --reverse              Reverse sort to ascending. Must be combined with --sort
      --sort string          Sort descending by field (look at JSON output for field names)
  -s, --source strings       Display events matching supplied sources
  -t, --sourcetype strings   Display events matching supplied sourcetypes
```

### extract
---
Outputs `ldscope`, `libscope.so`, and `scope.yml` to the provided directory. You can configure these files to instrument any application, and to output the data to any existing tool via simple TCP protocols.

#### Usage

<!-- Do we have the arguments in correct order? John's notes reverse them. -->

  `scope extract [flags] (<dir>)`

#### Aliases:
  `extract`, `excrete`, `expunge`, `extricate`, `exorcise`

#### Examples:

```
scope extract
scope extract /opt/libscope
scope extract --metricdest tcp://some.host:8125 --eventdest tcp://other.host:10070 .
```

#### Flags:
 
```
  -a, --authtoken string      Set AuthToken for Cribl Stream
  -c, --cribldest string      Set Cribl Stream destination for metrics & events (host:port defaults to tls://)
  -e, --eventdest string      Set destination for events (host:port defaults to tls://)
  -h, --help                  Help for extract
  -m, --metricdest string     Set destination for metrics (host:port defaults to tls://)
      --metricformat string   Set format of metrics output (statsd|ndjson); default is "ndjson"
  -n, --nobreaker             Set Cribl Stream to not break streams into events
```

### flows
---

Displays observed flows from the given session. If run with payload capture on, outputs full payloads from the flow.

#### Usage

<!-- How should we notate ID arguments? see also scope events -->

`scope flows [flags] <sessionId>`

#### Examples

```
scope flows                # Displays all flows
scope flows 124x3c         # Displays more info about the flow
scope flows --in 124x3c    # Displays the inbound payload of that flow
scope flows --out 124x3c   # Displays the outbound payload of that flow
scope flows -p 0.0.0.0/24  # Displays flows in that subnet range
scope flows --sort net_host_port --reverse  # Sort flows by ascending host port
```

#### Flags

```
  -a, --all           Show all flows
  -h, --help          Help for flows
  -i, --id int        Display flows from specific from session ID (default -1)
      --in            Output contents of the inbound payload. Requires flow ID specified.
  -j, --json          Output as newline-delimited JSON
  -n, --last int      Show last <n> flows (default 20)
      --out           Output contents of the outbound payload. Requires flow ID specified.
  -p, --peer ipNet    Filter to peers in the given network
  -r, --reverse       Reverse sort to ascending
  -s, --sort string   Sort descending by field (look at JSON output for field names)
  
```

### help
---

Displays help content for any AppScope subcommand. Just type `scope help [subcommand]` for full details.

#### Usage

`scope help [subcommand]`

#### Examples

`scope help run`

### history
---

Prints information about sessions. Every time you scope a command, that is called an AppScope session. Each session has a directory which is referenced by a session ID. By default, the AppScope CLI stores all the information it collects during a given session in that session's directory. When you run `history`, you see a listing of sessions, one session per scoped command, along with information about when the session started, how many events were output during the session, and so on.

#### Usage

`scope history [flags]`

#### Aliases

`history, hist`

#### Examples

```
scope history                    # Displays session history
scope hist                       # Shortcut for scope history
scope hist -r                    # Displays running sessions
scope hist --id 2                # Displays detailed information for session 2
scope hist -n 50                 # Displays last 50 sessions
scope hist -d                    # Displays directory for the last session
cat $(scope hist -d)/args.json   # Outputs contents of args.json in the scope history directory for the current session
```

#### Flags

```
  -a, --all        List all sessions
  -d, --dir        Output just directory (with -i)
  -h, --help       Help for history
  -i, --id int     Display info from specific from session ID (default -1)
  -n, --last int   Show last <n> sessions (default 20)
  -r, --running    List running sessions
```

### k8s
---

Prints configurations to pass to `kubectl`, which then automatically instruments newly-launched containers. This installs a mutating admission webhook, which adds an `initContainer` to each pod. The webhook also sets environment variables that install AppScope for all processes in that container.

The `--*dest` flags accept file names like `/tmp/scope.log`; URLs like `file:///tmp/scope.log`; or sockets specified with the pattern `unix:///var/run/mysock`, `tcp://hostname:port`, `udp://hostname:port`, or `tls://hostname:port`.

#### Usage

`scope k8s [flags]`

#### Examples

```
scope k8s --metricdest tcp://some.host:8125 --eventdest tcp://other.host:10070 | kubectl apply -f -
kubectl label namespace default scope=enabled
```

#### Flags

```
      --app string            Name of the app in Kubernetes (default "scope")
  -a, --authtoken string      Set AuthToken for Cribl Stream
      --certfile string       Certificate file for TLS in the container (mounted secret) (default "/etc/certs/tls.crt")
  -c, --cribldest string      Set Cribl Stream destination for metrics & events (host:port defaults to tls://)
      --debug                 Turn on debug logging in the scope webhook container
  -e, --eventdest string      Set destination for events (host:port defaults to tls://)
  -h, --help                  Help for k8s
      --keyfile string        Private key file for TLS in the container (mounted secret) (default "/etc/certs/tls.key")
  -m, --metricdest string     Set destination for metrics (host:port defaults to tls://)
      --metricformat string   Set format of metrics output (statsd|ndjson); default is "ndjson"
      --namespace string      Name of the namespace in which to install; default is "default"
  -n, --nobreaker             Set Cribl Stream to not break streams into events
      --port int              Port to listen on (default 4443)
      --server                Run Webhook server
      --version string        Version of scope to deploy

```

### logs
---

Displays internal AppScope logs for troubleshooting AppScope itself.

#### Usage

`scope logs [flags]`

#### Examples

```
scope logs
```

#### Flags

```
  -h, --help             Help for logs
  -i, --id int           Display logs from specific from session ID (default -1)
  -n, --last int         Show last <n> lines (default 20)
  -s, --scope            Show scope.log (from CLI) instead of ldscope.log (from library)
  -S, --service string   Display logs from a systemd service instead of a session)
 
```

### metrics
---

Outputs metrics for a session.

#### Usage

`scope metrics [flags]`

#### Examples

`scope metrics`

#### Flags

```
  -c, --cols             Display metrics as columns. Must be combined with -m
  -g, --graph            Graph this metric. Must be combined with -m
  -h, --help             Help for metrics
  -i, --id int           Display info from specific from session ID (default -1)
  -m, --metric strings   Display for specified metrics only (comma-separated)
  -u, --uniq             Display first instance of each unique metric
```

### prune
----

Prunes (deletes) one or more sessions from the history.

#### Usage

`scope prune [flags]`

#### Examples

```
scope prune -k 20
scope prune -a
scope prune -d 1
```

#### Flags

Negative arguments are not allowed.

```
  -a, --all          Delete all sessions
  -d, --delete int   Delete last <delete> sessions
  -f, --force        Do not prompt for confirmation
  -h, --help         Help for prune
  -k, --keep int     Keep last <keep> sessions, delete all others
```

### ps
---

Lists all processes into which the libscope library is injected.

#### Usage

`scope ps`

### run
----

Executes a scoped command. By default, calling `scope` with no subcommands will run the executables you pass as arguments to 
`scope`. However, `scope` allows for additional arguments to be passed to `run`, to capture payloads or to increase metrics' 
verbosity. Must be called with the `--` flag, e.g., `scope run -- <command>`, to prevent AppScope from attempting to parse flags passed to the executed command.

The `--*dest` flags accept file names like `/tmp/scope.log`; URLs like `file:///tmp/scope.log`; or sockets specified with the pattern `unix:///var/run/mysock`, `tcp://hostname:port`, `udp://hostname:port`, or `tls://hostname:port`.

#### Usage

`scope run [flags] [command]`

#### Examples

```
scope run -- /bin/echo "foo"
scope run -- perl -e 'print "foo\n"'
scope run --payloads -- nc -lp 10001
scope run -- curl https://wttr.in/94105
scope run -c tcp://127.0.0.1:10091 -- curl https://wttr.in/94105
scope run -c edge -- top
```

#### Flags

```
  -a, --authtoken string      Set AuthToken for Cribl Stream
  -c, --cribldest string      Set Cribl Stream destination for metrics & events (host:port defaults to tls://)
  -e, --eventdest string      Set destination for events (host:port defaults to tls://)
  -h, --help                  Help for run
  -l, --librarypath string    Set path for dynamic libraries
      --loglevel string       Set ldscope log level (debug, warning, info, error, none)
  -m, --metricdest string     Set destination for metrics (host:port defaults to tls://)
      --metricformat string   Set format of metrics output (statsd|ndjson); default is "ndjson"
  -n, --nobreaker             Set Cribl Stream to not break streams into events
      --passthrough           Run ldscope with current environment & no config
  -p, --payloads              Capture payloads of network transactions
  -u, --userconfig string     Run ldscope with a user specified config file; overrides all other settings
  -v, --verbosity int         Set scope metric verbosity (default 4)

```

### service
---

Configures the specified `systemd`/`OpenRC` service to be scoped upon starting.

#### Usage

`scope service SERVICE [flags]`

#### Examples

```
scope service cribl -c tls://in.my-instance.cribl.cloud:10090
```

#### Flags

```
  -a, --authtoken string      Set AuthToken for Cribl Stream
  -c, --cribldest string      Set Cribl Stream destination for metrics & events (host:port defaults to tls://)
  -e, --eventdest string      Set destination for events (host:port defaults to tls://)
      --force                 Bypass confirmation prompt
  -h, --help                  Help for service
  -m, --metricdest string     Set destination for metrics (host:port defaults to tls://)
      --metricformat string   Set format of metrics output (statsd|ndjson); default is "ndjson"
  -n, --nobreaker             Set Cribl Stream to not break streams into events
  -u, --user string           Specify owner username
  
```

### start
---

Start scoping a filtered selection of processes and services on the host and in all relevant containers.

The following actions will be performed on the host and in all relevant containers:
- Extraction of libscope.so to /usr/lib/appscope/<version>/ or /tmp/appscope/<version>/
- Extraction of the filter input to /usr/lib/appscope/scope_filter or /tmp/appscope/scope_filter
- Attach to all existing "allowed" processes defined in the filter file
- Install etc/profile.d/scope.sh script to preload /usr/lib/appscope/<version>/libscope.so if it exists
- Modify the relevant service configurations to preload /usr/lib/appscope/<version>/libscope.so or /tmp/appscope/<version>/libscope.so

#### Usage

`scope start [flags]`

#### Examples

```
scope start < example_filter.yml
cat example_filter.json | scope start
```

#### Flags

```
  -f, --force   Use this flag when you're sure you want to run scope start
  -h, --help    help for start
```

### version
----

Outputs version info.

#### Usage

`scope version [flags]`

#### Examples

```
scope version
scope version --date
scope version --summary
scope version --tag
```

#### Flags

```
      --date      Output just the date
  -h, --help      Help for version
      --summary   Output just the summary
      --tag       Output just the tag
```

### watch
---

Executes a scoped command on an interval. Must be called with the `--` flag, e.g., `scope watch -- <command>`, to prevent AppScope from attempting to parse flags passed to the executed command.

#### Usage

`scope watch [flags]`

#### Examples

```
scope watch -i 5s -- /bin/echo "foo"
scope watch --interval=1m-- perl -e 'print "foo\n"'
scope watch --interval=5s --payloads -- nc -lp 10001
scope watch -i 1h -- curl https://wttr.in/94105
scope watch --interval=10s -- curl https://wttr.in/94105
```

#### Flags

```
  -a, --authtoken string      Set AuthToken for Cribl Stream
  -c, --cribldest string      Set Cribl Stream destination for metrics & events (host:port defaults to tls://)
  -e, --eventdest string      Set destination for events (host:port defaults to tls://)
  -h, --help                  Help for watch
  -i, --interval string       Run every <x>(s|m|h)
  -l, --librarypath string    Set path for dynamic libraries
      --loglevel string       Set ldscope log level (debug, warning, info, error, none)
  -m, --metricdest string     Set destination for metrics (host:port defaults to tls://)
      --metricformat string   Set format of metrics output (statsd|ndjson); default is "ndjson"
  -n, --nobreaker             Set Cribl Stream to not break streams into events
      --passthrough           Run ldscope with current environment & no config
  -p, --payloads              Capture payloads of network transactions
  -u, --userconfig string     Run ldscope with a user specified config file. Overrides all other settings.
  -v, --verbosity int         Set scope metric verbosity (default 4)
  
```

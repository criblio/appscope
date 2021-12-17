---
title: CLI Reference
---

## CLI Reference
---

### Command Syntax

To execute CLI commands, the basic syntax is:

```
./scope <command> [flags] [options]
```

### Commands Available

To see a list of available commands, enter `./scope` alone, or `./scope -h`, or `./scope --help`. This displays the basic help listing below.

```
Cribl AppScope Command Line Interface

AppScope is a general-purpose observable application telemetry system.

Running `scope` with no subcommands will execute the `scope run` command.

Usage:
  scope [command]

Available Commands:
  attach      Scope an existing PID
  completion  Generate completion code for specified shell
  dash        Display scope dashboard
  events      Outputs events for a session
  extract     Output instrumentary library files to <dir>
  flows       Observed flows from the session, potentially including payloads
  help        Help about any command
  history     List scope session history
  k8s         Install scope in kubernetes
  logs        Display scope logs
  metrics     Outputs metrics for a session
  prune       Prune deletes session history
  ps          List processes currently being scoped
  run         Executes a scoped command
  service     Configure a systemd service to be scoped
  version     Display scope version
  watch       Executes a scoped command on an interval

Flags:
  -h, --help   help for scope

Use "scope [command] --help" for more information about a command.
```

As noted just above, to see a specific command's help or its required parameters, enter: 
`./scope <command> -h` 

…or: 
`./scope help <command> [flags]`.

### dash
---

Displays an interactive dashboard with an overview of what's happening with the selected session.

#### Usage

`scope dash [flags]`

#### Examples

`scope dash`

#### Flags

```
  -h, --help     help for dash
  -i, --id int   Display info from specific from session ID (default -1)
```

### events
---
Outputs events for a session. You can obtain detailed information about each event by inputting the Event ID as a positional parameter. (By default, the Event ID appears in blue, in `[]`'s at the left.) You can provide filters to narrow down by name (e.g., `http`, `net`, `fs`, or `console`), or by field (e.g., `fs.open`, `stdout`, or `net.conn.open`). You can use JavaScript expressions to further refine the query, and to express logic.

#### Usage

`scope events [flags] ([eventId])`

#### Examples

```
scope events
scope events -t http
scope events -s stderr
scope events -e 'sourcetype!="net"'
scope events -n 1000 -e 'sourcetype!="console" && source.indexOf("cribl.log") == -1 && (data["file.name"] || "").indexOf("/proc") == -1'
```

#### Flags

```
  -a, --all                  Show all events
      --allfields            Displaying hidden fields
      --color                Force color on (if tty detection fails or pipeing)
  -e, --eval string          Evaluate JavaScript expression against event. Must return truthy to print event.
  -f, --follow               Follow a file, like tail -f
  -h, --help                 help for events
  -i, --id int               Display info from specific from session ID (default -1)
  -j, --json                 Output as newline delimited JSON
  -n, --last int             Show last <n> events (default 20)
  -m, --match string         Display events containing supplied string
  -s, --source strings       Display events matching supplied sources
  -t, --sourcetype strings   Display events matching supplied sourcetypes
```


### extract
---
Outputs `ldscope`, `libscope.so`, `scope.yml`, and `scope_protocol.yml` to the provided directory. You can configure these files to instrument any application, and to output the data to any existing tool via simple TCP protocols. The `libscope` component can easily be used with any dynamic or static application, regardless of the runtime.

#### Usage
  `scope extract (<dir>) [flags]`

#### Aliases:
  `extract`, `excrete`, `expunge`, `extricate`, `exorcise`

#### Examples:

```
scope extract
scope extract /opt/libscope
```

#### Flags:
 
```
  -h, --help   help for extract
```


### history
---
Lists scope session history.

#### Usage

`scope history [flags]`

#### Aliases

`history, hist`

#### Examples

`scope history`

#### Flags

```
  -a, --all        List all sessions
  -d, --dir        Output just directory (with -i)
  -h, --help       help for history
  -i, --id int     Display info from specific from session ID (default -1)
  -n, --last int   Show last <n> sessions (default 20)
  -r, --running    List running sessions
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
  -c, --cols             Display metrics as columns
  -g, --graph            Graph this metric
  -h, --help             help for metrics
  -i, --id int           Display info from specific from session ID (default -1)
  -m, --metric strings   Display only supplied metrics
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
scope delete -d 1
```

#### Flags

Negative arguments are not allowed.

```
  -a, --all          Delete all sessions
  -d, --delete int   Delete last <delete> sessions
  -f, --force        Do not prompt for confirmation
  -h, --help         help for prune
  -k, --keep int     Keep last <keep> sessions, delete all others
```

### run
----

Executes a scoped command (or application.

#### Usage

`scope run [flags]`

#### Examples

`scope run /bin/echo "foo"`

#### Flags

```
  -h, --help            help for run
      --passthrough     Runs scopec with current environment & no config.
  -p, --payloads        Capture payloads of network transactions
  -v, --verbosity int   Set scope metric verbosity (default 4)
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
```

#### Flags

```
      --date      output just the date
  -h, --help      help for version
      --summary   output just the summary
```

### --verbose
----

This flag sets the verbosity level. `0` is least verbose, `4` is the default, and `9` is most verbose. For descriptions of individual levels, see [Config File](/docs/config-file).

#### Usage

```
scope --verbose <level>
scope -v <level>
```

#### Examples

```
scope --verbose <level>
scope -v <level>
```

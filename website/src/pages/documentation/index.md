---
title: Intro
---

**Cribl AppScope**

# Introduction

AppScope is an open source instrumentation utility for any application, regardless of programming language with no code modification required. It consists of a shared library and an executable. In simple terms, the library is loaded into the address space of a process and details are extracted from applications as they execute. The executable provides a command line interface (CLI) which can optionally be used to control which processes are interposed, what data is extracted and how to view and interact with the results.

Information is extracted by the library as it interposes functions. When an application calls a function it actually calls a function of the same name in the AppScope library. Details are extracted from the function and the original function is called. The application is not aware of this interposition. The control and mechanisms used to interpose functions is provided by the AppScope library. The library is loaded using a number of mechanisms depending on the type of executable. Child processes are created with the library present, if the library was present in the parent. In this manner, a single executable is able to start, daemonize and create any number of children all of which include interposed functions.

The executable `scope` is provided with AppScope. The `scope` executable provides a rich CLI that allows for exploration of application functionality and detailed interaction with the data extracted from the AppScope library. The `scope` CLI supports an extensive terminal based dashboard providing a summary of application details. Views of the data include raw formats exported from the library as well as formatted output independent of data type. A history of "scoped" applications is maintained enabling comparison and capture of full fidelity application detail.

# Requirements

AppScope supports Linux distros and any runtime environment support by Linux.

# Install and Build

## Linux

(Only linux distros with yum or apt-get package managers are supported)

Run these commands to do everything: checkout the scope codebase, install necessary build tools, build, and run tests.

git clone https://github.com/criblio/appscope.git
cd appscope
./scope_env.sh build

`./scope_env.sh build` is a good command to know, it does everything needed to get a brand new environment up and running, but can also be run on an existing environment to rebuild, and rerun tests. If desired, individual parts of `./scope_env.sh build` can be run from the appscope directory.

./install_build_tools.sh
make all
make test

And finally, to clean out files that were manually added or the result of running build commands:

git clean -f -d

# Run

In order to run any given process in a mode where detailed information about the process can be extracted, without changes to code or the executable, you simply need to run the process with one or more environment variables set. You can set the environment variables defined below as needed. Or, just start the executable with the scope wrapper. Here's how: From the top level Scope directory execute the command:

./scope run your_command params
Where "your_command" is any executable and "params" are any parameters needed for the executable.

# Developer notes

- The top-level Makefile checks to see what platform you are on and then includes the specific Makefile for the platform from the OS specific directory.
- The run script will determine which platform you are using and will set environment variables accordingly.

# Environment variables

## Linux dynamic loader

LD_PRELOAD
This is the variable that is needed in order for the Scope library to be loaded and enable functions in new processes to be interposed on Linux.

## Scope-specific

SCOPE*CONF_PATH
Directly specify location and name of config file. Used only at start-time.
SCOPE_HOME
Specify a directory from which conf/scope.yml or ./scope.yml can be found.
Used only at start-time only if SCOPE_CONF_PATH does not exist.
For more info see Config File Resolution below.
SCOPE_METRIC_VERBOSITY
0-9 are valid values. Default is 4. For more info see Verbosity below.
SCOPE_SUMMARY_PERIOD
Number of seconds between output summarizations. Default is 10
SCOPE_METRIC_DEST
Default is udp://localhost:8125. Format is one of: file:///tmp/output.log or
udp://Server:123
Server is servername or address. 123 is a port number or service name
SCOPE_STATSD_PREFIX
Specify a string to be prepended to every scope metric.
SCOPE_STATSD_MAXLEN
Default is 512
SCOPE_LOG_LEVEL
debug, info, warning, error, none. Default is error.
SCOPE_LOG_DEST
same format as SCOPE_METRIC_DEST above. Default is file:///tmp/scope.log
SCOPE_TAG*
Specify a tag to be applied to every metric.
Environment variable expansion is available e.g. SCOPE_TAG_user=$USER
SCOPE_CMD_DIR
Specifies a directory to look for dynamic configuration files.
See Dynamic Configuration below. Default is /tmp

# Features:

## Verbosity

Controls two different aspects of output - Tag Cardinality and Summarization.

###Tag Cardinality
0 No expanded statsd tags
1 adds 'data', 'unit'
2 adds 'class', 'proto'
3 adds 'op'
4 adds 'host', 'proc'
5 adds 'domain', 'file'
6 adds 'localip', 'remoteip', 'localp', 'port', 'remotep'
7 adds 'fd', 'pid'
8 adds 'duration'
9 adds 'numops'

### Summarization

0-4 has full event summarization
5 turns off 'error'
6 turns off 'filesystem open/close' and 'dns'
7 turns off 'filesystem stat' and 'network connect'
8 turns off 'filesystem seek'
9 turns off 'filesystem read/write' and 'network send/receive'

## Config File Resolution

If the SCOPE_CONF_PATH env variable is defined and the points to a file that can be opened, it will use this as the config file. If not, it searches for the config file in this priority order using the first it finds. If this fails too, it will look for scope.yml in the same directory as LD_PRELOAD.

$SCOPE_HOME/conf/scope.yml
$SCOPE_HOME/scope.yml
/etc/scope/scope.yml
~/conf/scope.yml
~/scope.yml
./conf/scope.yml
./scope.yml

## Dynamic Configuration

Dynamic Configuration allows configuration settings to be changed on the fly after process start-time. At every SCOPE_SUMMARY_PERIOD the library looks in SCOPE_CMD_DIR to see if a file scope.<pid> exists. If it exists, it processes every line, looking for environment variable-style commands. (e.g. SCOPE_CMD_DBG_PATH=/tmp/outfile.txt) It changes the configuration to match the new settings, and deletes the scope.<pid> file when it's complete.

```javascript{numberLines: true}
//This is how a markdown single line comment looks

plugins: [
  {
    resolve: `gatsby-transformer-remark`,
    options: {
      plugins: [
        `gatsby-remark-prismjs`,
      ]
    }
  }
]

const coolFunction = () => {
    return "Testing out the render here";
}
```

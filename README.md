# Cribl AppScope

## Introduction

AppScope is an open source instrumentation utility for any application, regardless of programming language, with no code modification required. 

AppScope extracts detailed information from applications as they execute, including all file operations, all network operations, and cleartext data from encrypted flows. AppScope enables you to extract all payloads in their entirety. You can capture any console I/O and updates to any log file, and can monitor CPU and memory resource usage in process detail.

## Architecture and Operation

AppScope consists of a shared library and an executable. In simple terms, you load the library into the address space of a process. With the library loaded, details are extracted from applications as they execute. The executable provides a command line interface (CLI), which can optionally be used to control which processes are interposed, what data is extracted, and how to view and interact with the results.

The library extracts information as it interposes functions. When an application calls a function, it actually calls a function of the same name in the AppScope library. AppScope extracts details from the function, and the original function call proceeds. This interposition of function calls requires no change to an application: it works with unmodified binaries, and its CPU and memory overhead are minimal. 

The library is loaded using a number of mechanisms, depending on the type of executable. A dynamic loader can preload the library (where supported) and AppScope is used to load static executables.

Child processes are created with the library present, if the library was present in the parent. In this manner, a single executable is able to start, daemonize, and create any number of children, all of which include interposed functions.

The executable `scope` is provided with AppScope. The `scope` executable provides a rich CLI that allows for exploration of application functionality, and for detailed interaction with the data extracted from the AppScope library. The `scope` CLI supports an extensive terminal-based dashboard, providing a summary of application details. Views of the data include raw formats exported from the library, as well as formatted output, independent of data type. A history of "scoped" applications is maintained, enabling comparison and capture of full-fidelity application detail.

## Requirements

### Operating System Support

AppScope runs on the following platforms:

- RedHat Enterprise Linux or CentOS 6.4 and later
- Ubuntu 16 and later
- Amazon Linux 1 and 2

### Resources

AppScope requires the following minimum resources:

- CPU:	x84-64 architecture 
- Memory:	1GB
- Disk:	20MB

### Known Limitations

AppScope does not support the following runtimes:

- Open JVM &lt; v.6
- Oracle JVM &lt; v.6
- Go &lt; v.1.8

## Build and Install

### Linux

We build on Ubuntu 18.04 currently and do not support other build environments at this time. The goal is to produce a "build-once, run-anywhere" executable. Consider building with [Docker](#docker) if you are running somthing other than Ubuntu 18.04.

Check out the code, install build tools, build, and run tests:

```
git clone https://github.com/criblio/appscope.git
cd appscope
./scope_env.sh build
```

`./scope_env.sh build` does everything needed to get a brand new environment up and running. It can also be run on an existing environment to rebuild, and rerun tests. If desired, individual parts of `./scope_env.sh build` can be run from the `appscope` directory:

```
./install_build_tools.sh
make all
make test
```

To clean out files that were manually added, or were added by build commands:

```
git clean -f -d
```

### Docker

If you have Docker installed on your local machine, you can skip installing the dependencies locally and instead use a container to build. Use the `docker/builder/Dockerfile` and the `make docker-build` target. That builds an `appscope-builder` image that is Ubuntu with the necessary dependencies installed. The target then runs the image, mounting the local directory into the container, and builds the project. You'll end up with the binary in `bin/linux/scope` assuming all goes well.

```
make docker-build
```

By default, the current versions of Ubuntu or Go are used in the container image. Setting the `IMAGE` and `GOLANG` build arguments can override these.  Use the `BUILD_ARGS` environment variable to pass extra arguments to `docker build`.  The example below forces the builder to use Go 1.16 and Ubuntu 21.04.

```
make docker-build BUILD_ARGS="--build-arg GOLANG=golang-1.16 --build-arg IMAGE=ubuntu:21.04"
```

See the `Makefile` for the `docker run` command we use. Run it yourself without the `--entrypoint` and `-c` arguments to get a shell in the builder to fiddle around yourself.

## Scoping a Program

There are multiple ways to "scope" an application (i.e., to enable AppScope to extract details from an application). Most users run `scope` when getting started and when manually investigating a program or system. We provide an alternative mechanism via the dynamic loader's `LD_PRELOAD` functionality for use by sysadmins to collect data from server processes and their children.

### Using `scope`

The `scope` executable provides a simple way of "scoping" an application:

```
scope your_command params
```

Where `your_command` is any executable and `params` are any parameters needed for the executable.

Using `scope` in this manner causes `your_command` to be run with the AppScope library loaded. The default configuration for `scope` is to emit all data to the local file system. The path for the data store is configurable. Run `scope` or `scope help` to get a list of commands and `scope help [command]` for details of a specific command.

```
$ scope help
Cribl AppScope Command Line Interface

AppScope is a general-purpose observable application telemetry system.

Running `scope` with no subcommands will execute the `scope run` command.

Usage:
  scope [command]

Available Commands:
  attach      Scope an existing PID
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
  run         Executes a scoped command
  version     Display scope version
  watch       Executes a scoped command on an interval

Flags:
  -h, --help   help for scope

Use "scope [command] --help" for more information about a command.
```

Here is a simple example of invoking the `scope` executable:

```
scope bash
```

This will cause the bash shell to be executed with the AppScope library. All subsequent commands executed from the shell will emit detailed data describing the behavior of the application.

Use the `scope attach [PID]` command to attach to a running process. This injects the AppScope library so the process starts emitting data based on the configuration options.

### Using the Dynamic Loader

The kernel runs the dynamic loader when an executable is started. The loader determines which executable to load, resolves dependencies, allocates memory, loads code, performs all required link operations, and starts the application. When resolving dependencies, the loader can be configured to load a specific library in addition to the library dependencies associated with the executable. Loading of an additional library is configured by setting the environment variable `LD_PRELOAD`.

The AppScope library (`libscope.so`) is embedded into the `scope` binary. Use the `scope extract [DIR]` command to extract it.

```
mkdir /opt/appscope
scope extract /opt/appscope
```

Now you can use it as shown below.

```
LD_PRELOAD=/opt/appscope/libscope.so curl https://cribl.io
```

This configures the loader to run `curl`, while preloading the additional library `libscope.so`. The result is that AppScope will emit details from the execution of `curl`. The specific data emitted, the format of the data, the transport for that data are all configurable. See the help output from `opt/appscope/ldscope --help` for details:

```
libscope.so all
```

From a shell prompt, you can export `LD_PRELOAD` after which any command run in the shell will be "scoped".

```
export LD_PRELOAD=./libscope.so
```

## Support & Feedback

End-user documentation built from the [website/](./website/) content here in the project is available at <https://appscope.dev/>. See the [/docs](http://appscope.dev/docs/) content there for more examples and details on how to use AppScope. Release notes and know issues are kept there currently.

Many of the developers are in the [Cribl Community](https://cribl.io/community/) on Slack. Use the `#appscope` channel there.

Please submit feature requests and defect reports to the project at <https://github.com/criblio/appscope>.

We are working to expand documentation and notes for developers and maintainers here in the [docs](./docs/) folder.

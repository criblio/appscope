**Cribl AppScope**

# Introduction
AppScope is an open source instrumentation utility for any application, regardless of programming language with no code modification required. It consists of a shared library and an executable.

AppScope extracts detailed information from applications as they execute, including all file operations, all network operations, clear text data from encrypted flows, all payloads can be extracted in their entirety, any console I/O is captured, updates to any log file is captured, CPU and memory resource usage are presented in process detail.

AppScope consists of a shared library and an executable. In simple terms, the library is loaded by either preload features supported by the dynamic loader or by a custom loader for static executables,  into the address space of a process, With the library loaded, and details are extracted from applications as they execute. The executable provides a command line interface (CLI) which can optionally be used to control which processes are interposed, what data is extracted, and how to view and interact with the results.

Information is extracted by the library as it interposes functions. When an application calls a function, it actually calls a function of the same name in the AppScope library. Details are extracted from the function, and the original function is called. The interposition of function calls in this manner does not require any change to an application, it works with unmodified binaries, while the CPU and memory overhead are minimal.

The control and mechanisms used to interpose functions are provided by the AppScope library. The library is loaded using a number of mechanisms, depending on the type of executable. Child processes are created with the library present, if the library was present in the parent. In this manner, a single executable is able to start, daemonize and create any number of children all of which include interposed functions. 

The executable `scope` is provided with AppScope. The `scope` executable provides a rich CLI that allows for exploration of application functionality and detailed interaction with the data extracted from the AppScope library. The `scope` CLI supports an extensive terminal based dashboard providing a summary of application details. Views of the data include raw formats exported from the library as well as formatted output independent of data type. A history of "scoped" applications is maintained enabling comparison and capture of full-fidelity application detail. 

# Requirements
## Operating System support
- RedHat Linux 6 and later
- CentOs 6 and later
- Ubuntu 16 and later

## Resources
- CPU:  	x84-64 architecture 
- Memory: 	2Gb
- Disk: 	12Mb


# Install and Build
## Linux

Check out the code, install build tools, build, and run tests.


    git clone https://github.com/criblio/appscope.git
    cd appscope
    ./scope_env.sh build

`./scope_env.sh build` does everything needed to get a brand new environment up and running. It can also be run on an existing environment to rebuild, and rerun tests.  If desired, individual parts of `./scope_env.sh build` can be run from the appscope directory.

    ./install_build_tools.sh
    make all
    make test

To clean out files that were manually added, or were added by the result of running build commands:

    git clean -f -d

# Run
There are two ways to "scope" an application; enable the extraction of details from an application using AppScope. One, use the `scope` executable and 2) use the preload capability of the dynamic loader. 

## Using scope
The `scope` executable provides a simple way of "scoping" an application:
```bash
    scope your_command params
```
Where "your_command" is any executable and "params" are any parameters needed for the executable.

Using `scope` in this manner causes your_command to be loaded as well as loading the AppScope library. The default configuration for `scope` is to emit all data to the local file system. The path for the data store is configurable. Check out `scope` help with a -h command line option for details.

```bash
    scope bash
```
This will cause the bash shell to be executed with the AppScope library. All subsequent command executed from the shell will emit detailed data describing the behavior of the application.

## Dynamic Loader
The dynamic loader is run by the kernel when an executable is started. The loader determines which executable to load, resolves dependencies, allocates memory, loads code, performs all required link operations and starts the application. When resolving dependencies the loader can be configured to load a specific library in addition to the library dependencies associated with the executable. Loading of an additional library is configured by setting the environment variable LD_PRELOAD.

Examples:
```bash
    LD_PRELOAD=./libscope.so curl https://cribl.io
```
This will configure the loader to load and run `curl` while loading the additional library libscope.so. The result is that details from the execution of `curl` will be emitted by AppScope. The specific data emitted, the format of the data, the transport for that data are all configurable. Reference the help menu with the libscope.so library:
```bash
    libscope.so all
```

From a bash shell prompt:
```bash
    export LD_PRELOAD=./libscope.so
```
This will set the LD_PRELOAD environment variable for all commands that execute in the current shell. The result is that all commands executed from the shell will cause libscope.so to be loaded and details of those applications emitted. 

# The `scope` Executable
`scope` provides a rich set of features for exploring data from applications that have been "scoped". 
```bash
    scope -h
```
Provides help for `scope`.

There are several distinct modes supported by `scope`:
- Run an application
- Explore events from "scoped" applications
- Explore metrics from "scoped" applications
- Manage history and the accumulated data set



# Developer notes
- The top-level Makefile checks to see what platform you are on and then includes the specific Makefile for the platform from the OS specific directory.
- The run script will determine which platform you are using and will set environment variables accordingly. 


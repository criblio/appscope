---
title: How AppScope Works
---

## AppScope Components
---

AppScope is a single binary download that consists of three components:

### The Command Line Interface (CLI)

The AppScope CLI (`scope`) provides a quick and convenient way to explore capabilities and to obtain insight into application behavior. No installation or configuration is required to get started exploring application 
behavior.

<span>&#x2605;&#x2605;&#x2605;</span> **WHEN TO USE**: The easiest way to get started with AppScope is to use the CLI. This provides a rich set of capabilities intended to capture data from single applications. Data is captured in the local file system and is managed through the CLI.


### The Loader

Linux, like many other operating systems, provides a loader capable of loading and linking dynamic and static executables as needed. AppScope [provides](/documentation/loader-library) a very simple component (`ldscope`) that supports loading static executables. This allows AppScope to interpose functions, and to thereby expose the same data that is exposed from dynamic executables. While the AppScope loader is optional for use with dynamic executables, it is required in order when used to extract details from a static executable.

<span>&#x2605;&#x2605;&#x2605;</span> **WHEN TO USE**: As you get more interested in obtaining details from  applications, explore using the AppScope library apart from the CLI. This allows for full configurability via the `scope.yml` configuration file available [here](/documentation/config-files). You can forward captured data details over UDP, TCP, and local or remote connections, and you can define specific events and metrics for export. You can use the AppScope loader to start applications that include the library.

### The Library

The AppScope library (`libscope`) is the [core component](/documentation/loader-library) that resides in application processes, extracting data as an application executes. You can closely configure the library's behavior using environment variables or a configuration file. 

The library extracts information by interposing functions. When an application calls a function, it actually calls a function of the same name in the `libscope` which extracts details from the function, and the original function call proceeds. This interposition of function calls requires no change to an application: it works with unmodified binaries, and its CPU and memory overhead are minimal.

The library is loaded using a number of mechanisms, depending on the type of executable. A dynamic loader can preload the library (where supported) and AppScope is used to load static executables.

Child processes are created with the library present, if the library was present in the parent. In this manner, a single executable is able to start, daemonize, and create any number of children, all of which include interposed functions.


<span>&#x2605;&#x2605;&#x2605;</span> **WHEN TO USE**: Similar to the loader above you can use the library independently using the same configuration [file](/documentation/config-files) and/or environment variables. Environment variables take precedence over the default configuration, as well as over details in any configuration file. You use an environment variable to load the library independent of any executable. Whether you use this option or the AppScope loader, you get full control of the data source, formats, and transports.


## System-Level Design 
---
![AppScope system-level design](./images/AppScope_SysLvlDesign.png)


### Performance Overhead
AppScope collects data with around 2% CPU overhead and **minimal** latency penalty. In most cases, the overhead is markedly less than ~2% which is significantly more efficient than legacy and current monitoring systems.

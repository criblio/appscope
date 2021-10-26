---
title: How AppScope Works
---

## How AppScope Works

AppScope extracts information by interposing functions. When an application calls a function, it actually calls a function of the same name in the AppScope library (`libscope`), which extracts details from the function. Then the original function call proceeds. This interposition of function calls requires no change to an application: it works with unmodified binaries, and its CPU and memory overhead are minimal.

Child processes are created with the AppScope library present, if the library was present in the parent. In this manner, a single executable is able to start, daemonize, and create any number of children, all of which include interposed functions.

### Performance Overhead

AppScope collects data with around 2% CPU overhead and **minimal** latency penalty. In most cases, the overhead is markedly less than 2%, which is significantly more efficient than legacy and current monitoring systems.

## System-level Design

The AppScope library (`libscope`) is the core component that resides in application processes, extracting data as an application executes. You can closely configure the library's behavior, using environment variables or a [configuration file](/docs/config-file). 

When you work with AppScope, you use the library either directly, through the `ldscope` utility, or through the AppScope CLI.

![AppScope system-level design](./images/AppScope_SysLvlDesign.png)

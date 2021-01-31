---
title: AppScope Components
---

# AppScope Components

AppScope consists of three components:

1. Command Line Interface

The AppScope CLI (`scope`) provides a quick and easy way to explore capabilities and to obtain insight into application behavior. Here, no installation or configuration is required to get started exploring application behavior.

2. Loader

Linux, like other operating systems, provides a loader capable of loading and linking dynamic and static executables as needed. AppScope provides a very simple component (`ldscope`) that supports loading static executables. This allows AppScope to interpose functions, and to thereby expose the same data that is exposed from dynamic executables. (While the AppScope loader is optional for use with dynamic executables, it is required in order when used to extract details from a static executable.)

3. Library

The AppScope library (`libscope`) is the core component that resides in application processes, extracting data as an application executes. You can closely configure the library's behavior using environment variables or a configuration file.

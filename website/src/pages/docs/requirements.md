---
title: Requirements
---

## Requirements

Requirements for AppScope are as follows:

### Operating Systems (Linux Only)

AppScope runs on:

- Ubuntu 16 and later
- Alpine and other distributions based on musl libc
- RedHat Enterprise Linux or CentOS 7 and later
- Amazon Linux 1 and 2
- Debian

When building AppScope from source, use:

- Ubuntu 18.04

This restriction is imposed to make the resulting executable more portable.

### System

- CPU: x86-64 and ARM64 architectures
- Memory: 1GB
- Disk: 20MB (library + CLI)

### Filesystem Configuration

The distros that AppScope supports all require the use of `/tmp`, `/dev/shm`, and `/proc`. You should avoid custom filesystem configuration that interferes with AppScope's ability to use these directories.

### Cross-Compatibility with Cribl Suite

AppScope 1.2, Cribl Stream 4.0, Cribl Edge 4.0, and Cribl Search 1.0 are mutually compatible. If you integrate any of these products with earlier versions of peer products, some or all features will be unavailable.

### Known Limitations

**Only** these runtimes are supported: 

- Open JVM 7 and later, Oracle JVM 7 and later, go1.9 through go1.19.

AppScope cannot:

- Unload the libscope library, once loaded.
- Instrument static executables that are not written in Go.
- Instrument Go executables on ARM.
- Attach to any static application.

When an executable that's being scoped has been [stripped](https://en.wikipedia.org/wiki/Strip_(Unix)), it is not possible for `libscope.so` to obtain a file descriptor for an SSL session, and in turn, AppScope cannot include IP and port number fields in HTTP events.

Static executables can be scoped only if they are written in Go.

Static stripped Go executables can be scoped only when built with go1.13 or newer.

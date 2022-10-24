---
title: Known Issues
---

# Known Issues

## AppScope 1.2.0

2022-11-09 - Feature Release

As of this AppScope release, known issues include:

- [#1055](https://github.com/criblio/appscope/issues/1055) With certain combinations of mount and PID namespaces, AppScope fails to attach to a process. We have found two scenarios when attach will fail. One is when AppScope is on a host and the target process is in a container, making the mount namespaces different, but the process was started with `pid=host`, making the PID namespaces the same. Another is when both AppScope and the target process are on a host, making the PID namespaces the same, but the process is [configured](https://www.freedesktop.org/software/systemd/man/systemd.exec.html#PrivateTmp=) as a service with `PrivateTmp` set to `true`, making the mount namespaces different.
  - **Fix:** Planned for 1.2.1

## AppScope 1.1.0

2022-06-29 - Maintenance Release

As of this AppScope release, known issues include:

- [#1017](https://github.com/criblio/appscope/issues/1017) **Updated description**: AppScope incorrectly handles the "peek" flag in interposed functions that receive network data. When a server "peeks" at the first byte of data, AppScope counts that byte twice, which breaks protocol detection. AppScope then fails to correctly produce HTTP events.
  - **Fix:** 1.1.1, in [#1018](https://github.com/criblio/appscope/issues/1018)

## AppScope 1.0.3

2022-04-12 - Maintenance Release

As of this AppScope release, known issues include:

- [#869](https://github.com/criblio/appscope/issues/869) AppScope's ability to avoid outputting binary data in console messages does not work perfectly in all situations. When data is first sent to the console by a scoped app, AppScope can detect whether it is binary, and if so, redact it while emitting an informative message. But when apps switch back and forth between textual and binary data, AppScope does not detect the change, and binary data can appear in console output.
  - **Fix:** 1.0.4
  
- [#831](https://github.com/criblio/appscope/issues/831) When you scope the Microsoft Edge browser, AppScope fails to collect some HTTP events in these situations: (1) When Edge requests the use of HTTP/3; (2) When Edge uses HTTP/1.1 over UDP to request a list of plug-and-play devices from a Microsoft service; and, (3) when Edge uses SSL.  

## AppScope 1.0.0

2022-02-15 - GA

As of this AppScope GA release, known issues include:

- [#405](https://github.com/criblio/appscope/issues/405) `scope attach` to GDB seems to break GDB.

## AppScope 0.8.1

2021-12-21 - Maintenance Pre-Release

As of this AppScope pre-release, known issues include:

- [#677](https://github.com/criblio/appscope/issues/677) HTTP/1.1 events are parsed incorrectly, populating the `http_method` and `http_target` fields with junk.

  - **Fix:** 1.0.0, in [#680](https://github.com/criblio/appscope/issues/680)  

## AppScope 0.7

2021-07-02 - Maintenance Pre-Release

As of this AppScope pre-release, known issues include:

- [#362](https://github.com/criblio/appscope/issues/362) AppScope cannot attach to any musl libc process.

  - **Fix:** 0.7.1, in [#378](https://github.com/criblio/appscope/issues/378)  

- [#370](https://github.com/criblio/appscope/issues/370) Unattach/detach, and unload of the libscope library, are not supported.

## AppScope 0.6

2021-04-01 - Maintenance Pre-Release

As of this AppScope pre-release, known issues include:

- AppScope supports all Java [currently supported runtimes](https://dev.java/download/releases/): versions 7, 8, 11, and 17. It does not support JRE 6 or earlier.

- [#640](https://github.com/criblio/appscope/issues/640) On Alpine and other distributions based on musl libc, running `appscope attach` against a Go application fails with the error: `failed to find libc in target process`.

  - **Fix:** 1.0.0, in [#694](https://github.com/criblio/appscope/issues/694)

- [#35](https://github.com/criblio/appscope/issues/35) Go < v.1.8 is not supported.

- [#35](https://github.com/criblio/appscope/issues/35) Go executables with stripped symbols do not allow function interposition. Executables will run, but will not provide data to AppScope.
  - **Fix:** 0.7

- [#143](https://github.com/criblio/appscope/issues/143) Linux distros that use uClibc or MUSL instead of glibc, such as Alpine, are not supported
  - **Fix:** 0.7

- [#10](https://github.com/criblio/appscope/issues/10), [#165,](https://github.com/criblio/appscope/issues/165) Scoping the Docker executable is problematic.
  - **Fix:** 0.7

- [#119](https://github.com/criblio/appscope/issues/119) HTTP 2 metrics and headers can be viewed only with [Cribl LogStream](https://cribl.io/product/).

  - **Fix:** 0.8.0, in [#543](https://github.com/criblio/appscope/issues/543) 

<hr>

See [all issues](https://github.com/criblio/appscope/issues) currently open on GitHub.
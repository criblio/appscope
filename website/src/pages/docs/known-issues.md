---
title: Known Issues
---

# Known Issues

## AppScope 0.8.1

2021-12-21 - Maintenance Pre-Release

As of this AppScope pre-release, known issues include:

- [#677](https://github.com/criblio/appscope/issues/677) HTTP/1.1 events are parsed incorrectly, populating the `http_method` and `http_target` fields with junk.

## AppScope 0.7

2021-07-02 - Maintenance Pre-Release

As of this AppScope pre-release, known issues include:

- [#362](https://github.com/criblio/appscope/issues/362) AppScope cannot attach to any musl libc process.

- [#370](https://github.com/criblio/appscope/issues/370) Unattach/detach, and unload of the libscope library, are not supported.

## AppScope 0.6

2021-04-01 - Maintenance Pre-Release

As of this AppScope pre-release, known issues include:

- Java JRE < v.6 is not supported.

- [#35](https://github.com/criblio/appscope/issues/35) Go < v.1.8 is not supported.

- [#35](https://github.com/criblio/appscope/issues/35) Go executables with stripped symbols do not allow function interposition. Executables will run, but will not provide data to AppScope.
  - **Fix:** 0.7

- [#143](https://github.com/criblio/appscope/issues/143) Linux distros that use uClibc or MUSL instead of glibc, such as Alpine, are not supported
  - **Fix:** 0.7

- [#10](https://github.com/criblio/appscope/issues/10), [#165,](https://github.com/criblio/appscope/issues/165) Scoping the Docker executable is problematic.
  - **Fix:** 0.7

- [#119](https://github.com/criblio/appscope/issues/119) HTTP 2 metrics and headers can be viewed only with [Cribl LogStream](https://cribl.io/product/).

<hr>

See [all issues](https://github.com/criblio/appscope/issues) currently open on GitHub.
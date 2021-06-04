---
title: Changelog
---

# Changelog

## AppScope 0.7

2021-06-15 - Maintenance Pre-Release

This pre-release addresses the following issues:

- **Improvement**: [#267](https://github.com/criblio/appscope/issues/267),[#256](https://github.com/criblio/appscope/issues/256) Add support for attaching AppScope to a running process
- **Improvement**: [#292](https://github.com/criblio/appscope/issues/292) Add support for attaching AppScope to a running process from the CLI
- **Improvement**: [#286](https://github.com/criblio/appscope/issues/286) Add support for TLS over TCP connections, with new TLS-related environment variables shown by the command `ldscope --help configuration | grep TLS`
- **Improvement**: [#286](https://github.com/criblio/appscope/issues/286) Drop support for CentOS 6, because TLS support requires GLIBC_2.17 or newer, and CentOS 6 is stalled at GLIBC_2.12
- **Improvement**: [#132](https://github.com/criblio/appscope/issues/132) Add new `scope logs` command to view logs from the CLI
- **Improvement**: Add new `--interval` option to `scope watch` command to run AppScope at intervals, from the CLI
- **Improvement**: Make developer docs available at https://github.com/criblio/appscope/tree/master/docs
- **Fix**: [#265](https://github.com/criblio/appscope/issues/265) Re-direct bash malloc functions to glibc to avoid segfaults and memory allocation issues
- **Fix**: [#279](https://github.com/criblio/appscope/issues/279) Correct handling of pointers returned by gnutls, which previously could cause `git clone` to crash

## AppScope 0.6.1

2021-04-26 - Maintenance Pre-Release

This pre-release addresses the following issues:

- **Improvement**: [#238](https://github.com/criblio/appscope/issues/238) Resolve multiple issues with payloads from Node.js processes
- **Improvement**: [#257](https://github.com/criblio/appscope/issues/257) Add missing console/log events, first observed in Python 3
- **Improvement**: [#249](https://github.com/criblio/appscope/issues/249) Add support for Go apps built with `‑buildmode=pie`
 
## AppScope 0.6

2021-03-16 - Maintenance Pre-Release

This pre-release addresses the following issues:

- **Improvement**: [#99](https://github.com/criblio/appscope/issues/99) Enable augmenting HTTP events with HTTP header and payload information
- **Improvement**: [#38](https://github.com/criblio/appscope/issues/38) Disable (by default) TLS handshake info in payload capture
- **Improvement**: [#36](https://github.com/criblio/appscope/issues/36) Enable sending running scope instances a signal to reload fresh configuration
- **Improvement**: [#126](https://github.com/criblio/appscope/issues/126) Add Resolved IP Addresses list to a DNS response event
- **Improvement**: [#100](https://github.com/criblio/appscope/issues/100) 
Add `scope run` flags to facilitate sending scope data to third-party systems; also add a `scope k8s` command to facilitate installing a mutating admission webhook in a Kubernetes environment
- **Improvement**: [#149](https://github.com/criblio/appscope/issues/149) 
Apply tags set via environment variables to events, to match their application to metrics
- **Fix**: [#37](https://github.com/criblio/appscope/issues/37) Capture DNS requests' payloads
- **Fix**: [#2](https://github.com/criblio/appscope/issues/2) Enable scoping of scope
- **Fix**: [#35](https://github.com/criblio/appscope/issues/35) Improve error messaging and logging when symbols are stripped from Go executables

## AppScope 0.5.1

2021-02-05 - Maintenance Pre-Release

This pre-release addresses the following issues:

- **Fix**: [#96](https://github.com/criblio/appscope/issues/96) Corrects JSON corruption observed in non-US locales
- **Fix**: [#101](https://github.com/criblio/appscope/issues/101) Removes 10-second processing delay observed in short-lived processes
- **Fix**: [#25](https://github.com/criblio/appscope/issues/25) Manages contention of multiple processes writing to same data files

## AppScope 0.5

2021-02-05 - Initial Pre-Release

AppScope is now a thing. Its public repo is at https://github.com/criblio/appscope.
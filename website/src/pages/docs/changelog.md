---
title: Changelog
---

# Changelog

See the AppScope repo to view [all issues](https://github.com/criblio/appscope/issues).

## AppScope 0.7.1

2021-07-20 - Maintenance Pre-Release

This pre-release addresses the following issues:

- **Improvement**: [#363](https://github.com/criblio/appscope/pull/363) Add a new `scope ps` command to report on all processes into which the libscope library is loaded.
- **Improvement**: [#363](https://github.com/criblio/appscope/pull/363) Add support for attaching to a process by name (preserving existing attach by PID functionality).
- **Improvement**: [#304](https://github.com/criblio/appscope/issues/304) Add support for configuring an `authToken` to pass to LogStream as a header, using `scope run -a ${authToken}` in the CLI, or the `cribl` section of `scope.yml`.
- **Improvement**: [#368](https://github.com/criblio/appscope/issues/368) Add the new `-n` or `--nobreaker` option for configuring LogStream breaker behavior from AppScope. This sets the `SCOPE_CRIBL_NO_BREAKER` environment variable to `true` when you start scope with `run` or `attach`.
- **Improvement**: [#359](https://github.com/criblio/appscope/pull/359) Users who run scope as root now see the message `WARNING: Session history will be stored in $HOME/.scope/history and owned by root`.
- **Improvement**: [#278](https://github.com/criblio/appscope/pull/278) Integration tests are now enabled for PRs to the `master` and `release/*` branches.
- **Improvement**: [#381](https://github.com/criblio/appscope/pull/381) Commits to master now redeploy the website only when there are changes in the `website/` content.
- **Improvement**: [#350](https://github.com/criblio/appscope/issues/350) In `docker-build`, add support for:
  - Overriding the `build` command, for example: `make docker-build CMD="make coreall"`.
  - Passing `-u $(id -u):$(id -g)` to Docker so that the current user owns build results.
  - Using `.dockerignore` to omit unnecessary and potentially large items like `**/core`, `website/`, `test/testContainers`.

- **Fix**: [#364](https://github.com/criblio/appscope/issues/364) Console data is now received reliably when using TLS in both low-latency (all local) and relatively high-latency environments (LogStream cloud).
- **Fix**: [#394](https://github.com/criblio/appscope/pull/394) Improved AppScope's regex for determining whether a given filename represents a log file.
- **Fix**: [#309](https://github.com/criblio/appscope/issues/309) The `scope flows` command now works when `stdin` is a pipe.
- **Fix**: [#358](https://github.com/criblio/appscope/pull/358) The defaults for `scope prune -d` and `scope prune -k` are now set to `0`.
- **Fix**: [#388](https://github.com/criblio/appscope/issues/388) Ensure that all dimension names contain underscores (`_`) rather than periods (`.`).
- **Fix**: [#393](https://github.com/criblio/appscope/pull/393) The metrics `http.client.duration` and `http.server.duration` are now the correct type, `timer`.

# Changelog

## AppScope 0.7

2021-07-02 - Maintenance Pre-Release

This pre-release addresses the following issues:

- **Improvement**: [#267](https://github.com/criblio/appscope/issues/267),[#256](https://github.com/criblio/appscope/issues/256) Add support for attaching AppScope to a running process.
- **Improvement**: [#292](https://github.com/criblio/appscope/issues/292) Add support for attaching AppScope to a running process from the CLI.
- **Improvement**: [#143](https://github.com/criblio/appscope/issues/143) Add support for the Alpine Linux distribution, including support for musl libc (the compact glibc alternative on which Alpine and some other Linux distributions are based).
- **Improvement**: [#164](https://github.com/criblio/appscope/issues/164), [#286](https://github.com/criblio/appscope/issues/286) Add support for TLS over TCP connections, with new TLS-related environment variables shown by the command `ldscope --help configuration | grep TLS`.
- **Deprecation**: [#286](https://github.com/criblio/appscope/issues/286) Drop support for CentOS 6 (because TLS support requires glibc 2.17 or newer).
- **Improvement**: [#132](https://github.com/criblio/appscope/issues/132) Add new `scope logs` command to view logs from the CLI.
- **Improvement**: Add new `scope watch` command to run AppScope at intervals of seconds, minutes, or hours, from the CLI.
- **Improvement**: Make developer docs available at [https://github.com/criblio/appscope/tree/master/docs](https://github.com/criblio/appscope/tree/master/docs).
- **Fix**: [#265](https://github.com/criblio/appscope/issues/265) Re-direct bash malloc functions to glibc, to avoid segfaults and memory allocation issues.
- **Fix**: [#279](https://github.com/criblio/appscope/issues/279) Correct handling of pointers returned by gnutls, which otherwise could cause `git clone` to crash.

## AppScope 0.6.1

2021-04-26 - Maintenance Pre-Release

This pre-release addresses the following issues:

- **Improvement**: [#238](https://github.com/criblio/appscope/issues/238) Resolve multiple issues with payloads from Node.js processes.
- **Improvement**: [#257](https://github.com/criblio/appscope/issues/257) Add missing console/log events, first observed in Python 3.
- **Improvement**: [#249](https://github.com/criblio/appscope/issues/249) Add support for Go apps built with `‑buildmode=pie`.
 
## AppScope 0.6

2021-04-01 - Maintenance Pre-Release

This pre-release addresses the following issues:

- **Improvement**: [#99](https://github.com/criblio/appscope/issues/99) Enable augmenting HTTP events with HTTP header and payload information.
- **Improvement**: [#38](https://github.com/criblio/appscope/issues/38) Disable (by default) TLS handshake info in payload capture.
- **Improvement**: [#36](https://github.com/criblio/appscope/issues/36) Enable sending running scope instances a signal to reload fresh configuration.
- **Improvement**: [#126](https://github.com/criblio/appscope/issues/126) Add Resolved IP Addresses list to a DNS response event.
- **Improvement**: [#100](https://github.com/criblio/appscope/issues/100) 
Add `scope run` flags to facilitate sending scope data to third-party systems; also add a `scope k8s` command to facilitate installing a mutating admission webhook in a Kubernetes environment.
- **Improvement**: [#149](https://github.com/criblio/appscope/issues/149) 
Apply tags set via environment variables to events, to match their application to metrics.
- **Fix**: [#37](https://github.com/criblio/appscope/issues/37) Capture DNS requests' payloads.
- **Fix**: [#2](https://github.com/criblio/appscope/issues/2) Enable scoping of scope.
- **Fix**: [#35](https://github.com/criblio/appscope/issues/35) Improve error messaging and logging when symbols are stripped from Go executables.

## AppScope 0.5.1

2021-02-05 - Maintenance Pre-Release

This pre-release addresses the following issues:

- **Fix**: [#96](https://github.com/criblio/appscope/issues/96) Corrects JSON corruption observed in non-US locales.
- **Fix**: [#101](https://github.com/criblio/appscope/issues/101) Removes 10-second processing delay observed in short-lived processes.
- **Fix**: [#25](https://github.com/criblio/appscope/issues/25) Manages contention of multiple processes writing to same data files.

## AppScope 0.5

2021-02-05 - Initial Pre-Release

AppScope is now a thing. Its public repo is at https://github.com/criblio/appscope.
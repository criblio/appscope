---
title: Changelog
---

# Changelog

See the AppScope repo to view [all issues](https://github.com/criblio/appscope/issues).

## AppScope 0.8.0

2021-10-26 - Maintenance Pre-Release

- **Improvement**: [#543](https://github.com/criblio/appscope/issues/543) HTTP events are now all processed by the `libscope.so` library, not LogStream as was the case [previously](https://github.com/criblio/appscope/issues/311).

- **Improvement**: [#572](https://github.com/criblio/appscope/issues/572) The CLI now supports invoking a configuration file, using the syntax `scope run -u scope.yml -- foo` where `scope.yml` is the configuration file and `foo` is the command being scoped.

- **Improvement**: [#241](https://github.com/criblio/appscope/issues/241),[#271](https://github.com/criblio/appscope/issues/271),[#379](https://github.com/criblio/appscope/issues/379) The ARM64 limitations described in [#241](https://github.com/criblio/appscope/issues/241) have now been resolved and no longer apply.


- **Fix**: [#598](https://github.com/criblio/appscope/issues/598) Attempting to scope an executable that is Go version 1.17 or newer no longer causes AppScope to crash. AppScope does not support Go 1.17 yet, but now recognizes these executables and displays an informative message. 

- **Fix**: [#481](https://github.com/criblio/appscope/issues/481),[#575](https://github.com/criblio/appscope/issues/575) With musl libc-based distros, previous limitations on file system write events are now resolved, and no longer apply.

- **Fix**: [#397](https://github.com/criblio/appscope/issues/397),[#403](https://github.com/criblio/appscope/issues/403),[#567](https://github.com/criblio/appscope/issues/567),[#586](https://github.com/criblio/appscope/issues/586) With musl libc-based distros and Java, previous limitations on `scope attach` are now resolved and no longer apply.


## AppScope 0.7.5

2021-10-05 - Maintenance Pre-Release

- **Improvement**: [#462](https://github.com/criblio/appscope/issues/462), [#540](https://github.com/criblio/appscope/issues/540) Add support for sending event/metric output to a Unix domain socket.

- **Improvement**: [#492](https://github.com/criblio/appscope/issues/492) Add support for custom configuration with new `custom` section in `scope.yml`.  

- **Improvement**: [#489](https://github.com/criblio/appscope/issues/489) Streamline configuration file structure by eliminating `scope_protocol.yml` and merging its content into `scope.yml`.

- **Improvement**: [#503](https://github.com/criblio/appscope/issues/503) Promote the `tags` section of `config.yml` to top level, removing it from the `metric` > `format` section.  

## AppScope 0.7.4

2021-09-14 - Maintenance Pre-Release

- **Fix**: [#264](https://github.com/criblio/appscope/issues/264),[#525](https://github.com/criblio/appscope/issues/525) `scope k8s` now (1) supports specifying the namespace to install into, and (2) when the [ConfigMap](https://kubernetes.io/docs/concepts/configuration/configmap/) in that namespace is edited, automatically propagates the changes to all namespaces instrumented with the label `scope=enabled`.

## AppScope 0.7.3

2021-09-01 - Maintenance/Hotfix Pre-Release

- **Fix**: [#516](https://github.com/criblio/appscope/issues/516) Where scope'd processes use TLS, child processes no longer get stuck in fork calls.

- **Fix**: [#442](https://github.com/criblio/appscope/issues/442)
 Remove TLS exit handling code that is no longer needed.

- **Fix**: [#518](https://github.com/criblio/appscope/issues/518)
 Fix a build caching issue.

## AppScope 0.7.2

2021-08-17 - Maintenance Pre-Release

- **Improvement**: [#241](https://github.com/criblio/appscope/issues/241) Add experimental support for ARM64 architecture, with [limitations](https://github.com/criblio/appscope/issues/241#issuecomment-824428842).

- **Improvement**: [#401](https://github.com/criblio/appscope/issues/401) When `scope` attaches to a bash process, it now emits events for any child processes.

- **Improvement**: [#395](https://github.com/criblio/appscope/issues/395) Add a new environment variable, `SCOPE_CONNECT_TIMEOUT_SECS`, to configure wait behavior. This prevents events and metrics from being dropped when a scoped command exits quickly while using a connection with network latency.

- **Improvement**: [#311](https://github.com/criblio/appscope/issues/311) AppScope can now stream HTTP payloads to LogStream, whether the payloads originate as HTTP/1.1 or HTTP/2 traffic. LogStream then converts the payloads to HTTP events.

- **Fix**: [#407](https://github.com/criblio/appscope/issues/407) To prevent errors, `scope prune` now rejects negative numbers as arguments. Positive arguments implicitly mean "how many sessions back" to delete or keep.

- **Fix**: [#416](https://github.com/criblio/appscope/issues/416) Ensure that file descriptors opened by libscope are not closed by SSH, to avoid scenarios where processes operating over network connections can get stuck at 100% CPU utilization.

- **Fix**: [#417](https://github.com/criblio/appscope/issues/417) Ensure that AppScope performs exit handling only after SSL is finished. This avoids double free errors where the OpenSSL exit handler, being called while TLS is not enabled, frees memory that was not allocated.

- **Fix**: [#327](https://github.com/criblio/appscope/issues/327) Protocol-detect events now use the `net` event type, which prevents them from being dropped by periodic threads.

This pre-release addresses the following issues:

## AppScope 0.7.1

2021-07-20 - Maintenance Pre-Release

This pre-release addresses the following issues:

- **Improvement**: [#363](https://github.com/criblio/appscope/pull/363) Add a new `scope ps` command to report on all processes into which the libscope library is loaded.
- **Improvement**: [#363](https://github.com/criblio/appscope/pull/363) Add support for attaching to a process by name (preserving existing attach by PID functionality), for example: `scope attach NAME`.
- **Improvement**: [#304](https://github.com/criblio/appscope/issues/304) Add support for configuring an `authToken` to pass to LogStream as a header, using `scope run -a ${authToken}` in the CLI, or the `cribl` section of `scope.yml`.
- **Improvement**: [#368](https://github.com/criblio/appscope/issues/368) Add the new `-n` or `--nobreaker` option for configuring LogStream breaker behavior from AppScope. This option prevents LogStream from running Event Breakers on an AppScope Source. The `--nobreaker` option can only be used with LogStream starting with version 3.0.4.
- **Improvement**: [#359](https://github.com/criblio/appscope/pull/359) Users who run scope as root now see the message `WARNING: Session history will be stored in $HOME/.scope/history and owned by root`.
- **Improvement**: [#278](https://github.com/criblio/appscope/pull/278) Integration tests are now enabled for PRs to the `master` and `release/*` branches.
- **Improvement**: [#381](https://github.com/criblio/appscope/pull/381) Commits to master now redeploy the website only when there are changes in the `website/` content.
- **Improvement**: [#350](https://github.com/criblio/appscope/issues/350) In `docker-build`, add support for:
  - Overriding the `build` command, for example: `make docker-build CMD="make coreall"`.
  - Passing `-u $(id -u):$(id -g)` to Docker so that the current user owns build results.
  - Using `.dockerignore` to omit unnecessary and potentially large items like `**/core`, `website/`, `test/testContainers`.

- **Fix**: [#364](https://github.com/criblio/appscope/issues/364) Improved AppScope's delivery of pending events to a TLS server, upon process exit.
- **Fix**: [#394](https://github.com/criblio/appscope/pull/394) Improved AppScope's regex for determining whether a given filename represents a log file.
- **Fix**: [#309](https://github.com/criblio/appscope/issues/309) The `scope flows` command now works when `stdin` is a pipe.
- **Fix**: [#388](https://github.com/criblio/appscope/issues/388) Ensure that all dimension names contain underscores (`_`) rather than periods (`.`). The `http.status` and `http.target` dimensions have been corrected. This change is not backwards-compatible.
- **Fix**: [#393](https://github.com/criblio/appscope/pull/393) The metrics `http.client.duration` and `http.server.duration` are now the correct type, `timer`. This change is not backwards-compatible.
- **Fix**: [#358](https://github.com/criblio/appscope/pull/358) Cleaned up help for `-d` and `-k` options to `scope prune`.

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

---
title: Changelog
---

# Changelog

See the AppScope repo to view [all issues](https://github.com/criblio/appscope/issues).

## AppScope 1.0.3

2022-04-12 - Maintenance Release

Assets are available via Docker and the Cribl CDN at the links below.

- `Docker`: `cribl/scope:1.0.3`
- `x86`: [https://cdn.cribl.io/dl/scope/1.0.3/linux/x86_64/scope](https://cdn.cribl.io/dl/scope/1.0.3/linux/x86_64/scope)
- `ARM`: [https://cdn.cribl.io/dl/scope/1.0.3/linux/aarch64/scope](https://cdn.cribl.io/dl/scope/1.0.3/linux/aarch64/scope)

### Fixes

AppScope 1.0.3 improves the quality of results when scoping Go executables:

- [#738](https://github.com/criblio/appscope/issues/738) When scoping Go executables, AppScope no longer fails to collect certain events. This was accomplished by creating interpositions for two Go functions: `unlinkat` and `getdents`. AppScope already had the ability to interpose the equivalent functions in C, so this fix brings AppScope's interposition of Go functions into parity with its interposition of C functions.   
- [#864](https://github.com/criblio/appscope/issues/864) When scoping Go executables that call the `openat` function, AppScope no longer fails to collect `fs.open` events. This brings AppScope up to date with changes to `openat` in recent versions of Go.
- [#862](https://github.com/criblio/appscope/issues/862) When scoping dynamic Go executables, AppScope no longer falsely logs nonexistent errors.

## AppScope 1.0.2

2022-03-15 - Maintenance Release

Assets are available via Docker and the Cribl CDN at the links below.

- `Docker`: `cribl/scope:1.0.2`
- `x86`: [https://cdn.cribl.io/dl/scope/1.0.2/linux/x86_64/scope](https://cdn.cribl.io/dl/scope/1.0.2/linux/x86_64/scope)
- `ARM`: [https://cdn.cribl.io/dl/scope/1.0.2/linux/aarch64/scope](https://cdn.cribl.io/dl/scope/1.0.2/linux/aarch64/scope)

### New Features and Improvements

AppScope 1.0.2 introduces fine-grained control of `scope events` output:

- [#826](https://github.com/criblio/appscope/issues/826) When you do filtering or matching with `scope events`, AppScope now shows **all** results by default. (The filtering/matching options are `--sort`, `--match`, `--source`, and `--sourcetype`.) Alternatively, you can limit the number of events with the `--last` option, e.g., `scope events --last 20`.
  - When you are **not** filtering or matching, `scope events` shows the last 20 results by default. Use `--last` to specify a different limit, or `--all` to see all results.
- [#20](https://github.com/criblio/appscope/issues/20) You can now specify one or more field names, with `scope events --fields`, and the output will show only those field names and their values alongside each event. (If you want to restrict output to only those events which **contain** the selected fields, add the `--match` option.)
- [#813](https://github.com/criblio/appscope/issues/813) You can use the `--sort` option to sort by a top-level field (i.e., a top-level element of the event body). By default, this returns events sorted in descending order, both for numeric values (such as timestamps) and string values (such as the values for the `proc` field). To sort in ascending order, add the `--reverse` flag.

Another usability improvement applies to the CLI as a whole:

- [#611](https://github.com/criblio/appscope/issues/611) Given combinations of arguments that don't make sense, `scope` now responds with an error.

### Fixes

- [#825](https://github.com/criblio/appscope/issues/825) Output of `scope events` is now rendered on one line.
- [#728](https://github.com/criblio/appscope/issues/728) Added more check (`_chk`) functions to `libscope.so`, without which a scoped command could fail when that command had been compiled with a particular GCC toolchain.
- [#787](https://github.com/criblio/appscope/issues/787) Through improved HTTP pipelining support, AppScope now avoids some cases where the `http_method` field in HTTP events contained junk. This work continues in [#829](https://github.com/criblio/appscope/issues/829).
- [#800](https://github.com/criblio/appscope/issues/800) AppScope now checks for conditions which caused Java applications to crash when scoped, and avoids the crashes by applying fallback behaviors where necessary.

## AppScope 1.0.1

2022-02-28 - Update to GA Release 

Assets are available via Docker and the Cribl CDN at the links below.

- `Docker`: `cribl/scope:1.0.1`
- `x86`: [https://cdn.cribl.io/dl/scope/1.0.1/linux/x86_64/scope](https://cdn.cribl.io/dl/scope/1.0.1/linux/x86_64/scope)
- `ARM`: [https://cdn.cribl.io/dl/scope/1.0.1/linux/aarch64/scope](https://cdn.cribl.io/dl/scope/1.0.1/linux/aarch64/scope)

### Fixes

AppScope 1.0.1 updates event and metric names. The AppScope team strongly recommends using AppScope 1.0.1 and not 1.0.0.

#### Schema

- [#815](https://github.com/criblio/appscope/issues/815) Names of events and metrics are now complete, correct, and consistent with the original design intent.

## AppScope 1.0.0

2022-02-15 - GA Release <!-- Official v1 Release -->

This release has been replaced by AppScope 1.0.1.

### New Features and Improvements

AppScope's official v1 release includes advances in metric capture and forwarding, stabilization, formal schema definition and associated standardization, and enhanced filesystem events.

#### Instrumentation

- [#426](https://github.com/criblio/appscope/issues/426) You can now scope (attach to) a running process on ARM64.

#### Logging

- [#678](https://github.com/criblio/appscope/issues/678) Time zone is present in all logs.
- [#636](https://github.com/criblio/appscope/issues/636) The log level of the `missing uv_fileno` error is reduced to debug level.

#### Events and Metrics

- [#644](https://github.com/criblio/appscope/issues/644) Application-specific metrics can be captured.
- [#725](https://github.com/criblio/appscope/issues/725) Metrics and events are now [defined](https://github.com/criblio/appscope/tree/master/docs/schemas) in JSON Schema.
- [#723](https://github.com/criblio/appscope/issues/723) Metrics and events are now standardized. (Summary [here](https://github.com/criblio/.appscope/issues/712#issuecomment-1030234850).)
- [#699](https://github.com/criblio/appscope/issues/699) Added an option to enable events via configuration file entries.
- [#402](https://github.com/criblio/appscope/issues/402) Added support for readdir to `fs` events.
- [#709](https://github.com/criblio/appscope/issues/709) Added support for unlinks / file removals to `fs` events.
- [#162](https://github.com/criblio/appscope/issues/162) Added `net.open` and `net.close` metrics.

#### Data Routing

- [#697](https://github.com/criblio/appscope/issues/697) Transport defaults and priorities have changed. (Summary [here](https://github.com/criblio/appscope/issues/712#issuecomment-1030234850).) 
- [#661](https://github.com/criblio/appscope/issues/661) The process uid and gid are now present in the `cribl` connection header.
- [#670](https://github.com/criblio/appscope/issues/670), [#700](https://github.com/criblio/appscope/issues/700), [#707](https://github.com/criblio/appscope/issues/707)  Added an `edge` transport type, with its own default search path and CLI support.
- [#571](https://github.com/criblio/appscope/issues/571) Added support for a `unix://` destination in the CLI.

### Fixes

- [#640](https://github.com/criblio/appscope/issues/640) An error message is now generated when AppScope can't attach to a static executable.
- [#677](https://github.com/criblio/appscope/issues/677), [#687](https://github.com/criblio/appscope/issues/687) Fixed HTTP header extraction.
- [#737](https://github.com/criblio/appscope/issues/737) Fixed a seg fault on a busy Jenkins.
- [#232](https://github.com/criblio/appscope/issues/232) Fixed HTTP metric aggregation.
- [#657](https://github.com/criblio/appscope/issues/657) `apt-get` no longer hangs.

## AppScope 0.8.1

2021-12-21 - Maintenance Pre-Release

- **Improvement**: [#533](https://github.com/criblio/appscope/issues/533) AppScope logs now include dropped message and connection failure indicators, which make it easier to diagnose TCP/TLS connection issues, or to see when no data is flowing from a scoped process.

- **Improvement**: [#534](https://github.com/criblio/appscope/issues/534) AppScope log entries now include timestamps, e.g., `[2021-10-20 17:48:36.974]`. This timestamp format is ambiguous, and a fix is planned for AppScope 1.0.0. See [#678](https://github.com/criblio/appscope/issues/678). 

- **Improvement**: [#229](https://github.com/criblio/appscope/issues/229) When a process starts, the User Identifier (UID) and Group Identifier (GID) now appear in AppScope's `proc.start` metric and process start event (also known as TCP JSON connection header).

- **Fix**: [#650](https://github.com/criblio/appscope/issues/650) When scoping a process with TLS, and that process does not last long enough for AppScope to complete the TLS connection, AppScope no longer causes the scoped process to crash.

## AppScope 0.8.0

2021-10-26 - Maintenance Pre-Release

- **Improvement**: [#543](https://github.com/criblio/appscope/issues/543) HTTP events are now all processed by the `libscope.so` library, not LogStream as was the case [previously](https://github.com/criblio/appscope/issues/311).

- **Improvement**: [#572](https://github.com/criblio/appscope/issues/572) The CLI now supports invoking a configuration file, using the syntax `scope run -u scope.yml -- foo` where `scope.yml` is the configuration file and `foo` is the command being scoped.

- **Improvement**: [#241](https://github.com/criblio/appscope/issues/241),[#271](https://github.com/criblio/appscope/issues/271),[#379](https://github.com/criblio/appscope/issues/379) The ARM64 limitations described in [#241](https://github.com/criblio/appscope/issues/241) are resolved and no longer apply, except that Go executables remain unsupported on ARM64.


- **Fix**: [#598](https://github.com/criblio/appscope/issues/598) Attempting to scope an executable that is Go version 1.17 or newer no longer causes AppScope to crash. AppScope does not support Go 1.17 yet, but now recognizes these executables and displays an informative message. 

- **Fix**: [#481](https://github.com/criblio/appscope/issues/481),[#575](https://github.com/criblio/appscope/issues/575) With musl libc-based distros, previous limitations on filesystem write events are now resolved, and no longer apply.

- **Fix**: [#397](https://github.com/criblio/appscope/issues/397),[#403](https://github.com/criblio/appscope/issues/403),[#567](https://github.com/criblio/appscope/issues/567),[#586](https://github.com/criblio/appscope/issues/586) With musl libc-based distros and Java, previous limitations on `scope attach` are now resolved and no longer apply.


## AppScope 0.7.5

2021-10-05 - Maintenance Pre-Release

- **Improvement**: [#462](https://github.com/criblio/appscope/issues/462), [#540](https://github.com/criblio/appscope/issues/540) Add support for sending event/metric output to a Unix domain socket.

- **Improvement**: [#492](https://github.com/criblio/appscope/issues/492) Add support for custom configuration with new `custom` section in `scope.yml`.  

- **Improvement**: [#489](https://github.com/criblio/appscope/issues/489) Streamline configuration file structure by eliminating `scope_protocol.yml` and merging its content into `scope.yml`.

- **Improvement**: [#503](https://github.com/criblio/appscope/issues/503) Promote the `tags` section of `scope.yml` to top level, removing it from the `metric` > `format` section. This change is **not** backwards-compatible. 

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

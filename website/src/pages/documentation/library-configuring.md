---
title: Configure the Library
---

# Using the Library

To use the AppScope library independently of the CLI or loader, you rely on the `LD_PRELOAD` environment variable. This section provides several examples – all calling the system-level `ps` command – simply to show how the syntax works.

## LD_PRELOAD Environment Variable with a Single Command

Start with this basic example:

`LD_PRELOAD=./libscope.so ps -ef`

This executes the command `ps -ef`. But first, the OS' loader loads the AppScope library as part of loading and linking the ps executable.

Details of the ps application's execution are emitted to the configured transport, in the configured format. For configuration details, see [Configuring the Library](#configuring).

## LD_PRELOAD Environment Variable Extended Examples

These examples demonstrate using `LD_PRELOAD` with additional variables.

1. `LD_PRELOAD=./libscope.so SCOPE_METRIC_VERBOSITY=5 ps -ef`

This again executes the `ps` command using the AppScope library. But it also defines the verbosity for metric extraction as level 5. (This verbosity setting overrides any config-file setting, as well as the default value.)

2. `LD_PRELOAD=./libscope.so SCOPE_HOME=/etc/scope ps -ef`

This again executes the `ps` command using the AppScope library. But it also directs the library to use the config file `/etc/scope/scope.yml`.

3. `LD_PRELOAD=./libscope.so SCOPE_EVENT_DEST=tcp://localhost:9999 ps -ef`

This again executes the `ps` command using the AppScope library. But here, we also specify that events (as opposed to metrics) will be sent over a TCP connection to localhost, using port 9999. (This event destination setting overrides any config-file setting, as well as the default value.)

# <span id="configuring"> Configuring the Library </span>

For details on configuring the library, see AppScope online help's CONFIGURATION section. For the default settings in the sample `scope.yml` configuration file, see the [Appendix](/documentation/default-configuration), or see the most-recent file on [GitHub](https://github.com/criblio/appscope/blob/master/conf/scope.yml).

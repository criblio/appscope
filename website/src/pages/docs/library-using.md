---
title: Using the Library
---

## Using the Library (libscope.so)
----

To use the library independently of the CLI or loader, you rely on the `LD_PRELOAD` environment variable. This section provides several examples – all calling the system-level `ps` command – simply to show how the syntax works.

### LD_PRELOAD Environment Variable with a Single Command

Start with this basic example:

```
LD_PRELOAD=./libscope.so ps -ef
```

This executes the command `ps -ef`. But first, the OS' loader loads the AppScope library, as part of loading and linking the `ps` executable.

Details of the `ps` application's execution are emitted to the configured transport, in the configured format. For configuration details, see [Configuring the Library](#configuring) below.

### LD_PRELOAD Environment Variable – Extended Examples

These examples demonstrate using `LD_PRELOAD` with additional variables.

#### Example 1: 

```
LD_PRELOAD=./libscope.so SCOPE_METRIC_VERBOSITY=5 ps -ef
```

This again executes the `ps` command using the AppScope library. But it also defines the verbosity for metric extraction as level `5`. (This verbosity setting overrides any config-file setting, as well as the default value.)

#### Example 2:

```
LD_PRELOAD=./libscope.so SCOPE_HOME=/etc/scope ps -ef
```

This again executes the `ps` command using the AppScope library. But it also directs the library to use the config file `/etc/scope/scope.yml`.

#### Example 3: 

```
LD_PRELOAD=./libscope.so SCOPE_EVENT_DEST=tcp://localhost:9999 ps -ef
```

This again executes the `ps` command using the AppScope library. But here, we also specify that events (as opposed to metrics) will be sent over a TCP connection to localhost, using port `9999`. (This event destination setting overrides any config-file setting, as well as the default value.)


### <span id="configuring">Configuring the Library</span>

For a full list of library environment variables, execute: `./libscope.so all`

For the default settings in the sample `scope.yml` configuration file, see [Config Files](/docs/config-files), or inspect the most-recent file on [GitHub](https://github.com/criblio/appscope/blob/master/conf/scope.yml).

### <span id="lambda">Deploying the Library in a Lambda Function</span>

You can pull the libscope.so library into an AWS Lambda function as a [Lambda layer](https://docs.aws.amazon.com/lambda/latest/dg/configuration-layers.html), using these steps:

1. Run `scope extract`, e.g.:
    ```
    scope extract /opt/<your-layer>/
    ```

1. Set these three environment variables:
    ```
    LD_PRELOAD=/opt/<your-layer>/libscope.so
    SCOPE_EXEC_PATH=/opt/<your-layer>/ldscope
    SCOPE_CONF_PATH=/opt/<your-layer>/scope.yml
    ```

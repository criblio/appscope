---
title: Using the Library
---

## Using the Library (libscope.so)

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

This again executes the `ps` command using the AppScope library. But it also defines the verbosity for metric extraction as level `5`. (This verbosity setting overrides any config-file setting, as well as the default value.)

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

#### Example 4:

This adds AppScope to a `systemd` (boot-time) service. 

For purposes of the example, the service will be `httpd`, described by an `httpd.service` file which contains an `EnvironmentFile=/etc/sysconfig/httpd` entry.

##### Step 1

Extract the library to a new directory (`/opt/scope` in this example):

```
mkdir /opt/scope && cd /opt/scope
curl -Lo scope https://cdn.cribl.io/dl/scope/\
 $(curl -L https://cdn.cribl.io/dl/scope/latest)/linux/scope && \
 chmod 755 ./scope
./scope extract .
```

The result will be that the system uses `/opt/scope/scope.yml` to configure `libscope.so`.


##### Step 2

Add an `LD_PRELOAD` environment variable to the `systemd` config file.

In the `httpd.service` file, edit the `/etc/sysconfig/httpd` entry to include the following environment variables:

```
SCOPE_HOME=/opt/scope
LD_PRELOAD=/opt/scope/libscope.so
```

### <span id="configuring">Configuring the Library</span>

Use the `ldscope` command with the `--help` option to obtain the information you need:

- `ldscope --help` shows basic help.
- `ldscope --help all` shows the full set of help content. 
- `ldscope --help configuration` lists the full set of library environment variables.

For the default settings in the sample `scope.yml` configuration file, see [Config File](/docs/config-file), or inspect the most-recent file on [GitHub](https://github.com/criblio/appscope/blob/master/conf/scope.yml).

### <span id="lambda">Deploying the Library in an AWS Lambda Function</span>

You can interpose the libscope.so library into an AWS Lambda function as a [Lambda layer](https://docs.aws.amazon.com/lambda/latest/dg/configuration-layers.html), using these steps. By default, Lambda functions use `lib` as their `LD_LIBRARY_PATH`, which makes loading AppScope very easy.

Run `scope extract`, e.g.:

```
mkdir lib
scope extract ./lib
```

Or grab the bits directly from the website:

```
mkdir lib
curl -Ls https://cdn.cribl.io/dl/scope/$(curl -Ls https://cdn.cribl.io/dl/scope/latest)/linux/scope.tgz | tar zxf - -C lib --strip-components=1 
```

Modify the `scope.yml` configuration file as appropriate, then compress everything into a `.zip` file:

```
tar pvczf lambda_layer.zip lib/
```

Create a [Lambda layer](https://docs.aws.amazon.com/lambda/latest/dg/configuration-layers.html#configuration-layers-create), and associate the runtimes you want to use with AppScope in your Lambda functions. Upload the `lambda_layer.zip` file created in the previous step.

Add the custom layer to your Lambda function by selecting the previously created layer and version. 

#### Environment Variables

At a minimum, you must set the `LD_PRELOAD` environment variable in your Lambda configuration:

```
LD_PRELOAD=libscope.so
```

For static executables (like the Go runtime), set `SCOPE_EXEC_PATH` to run the [loader](/docs/how-works):
```
SCOPE_EXEC_PATH=/lib/ldscope
```

You must also tell AppScope where to deliver events. This can be accomplished by setting one of the following environment variables:

- `SCOPE_CONF_PATH=lib/scope.yml`
or:
- `SCOPE_EVENT_DEST=tcp://host:port`
or:
- `SCOPE_CRIBL=tcp://host:port`


### TBD Where does this fit Library

To use the library for the first time in a given environment, complete this quick procedure:

1. Download AppScope. 

2. Choose a directory in your environment that you want to serve as the AppScope home directory (`SCOPE_HOME`), i.e., the directory from which AppScope runs.

3. Extract (`scope extract`) the contents of the AppScope binary into the AppScope home directory.

For example, you could create an AppScope home directory called `assets`:

```
# mkdir assets && scope extract assets
Successfully extracted to assets.


# ll assets/
total 3404
drwxr-xr-x 2 root root    4096 Jan 31 22:32 ./
drwxr-xr-x 1 root root    4096 Jan 31 22:32 ../
-rwxr-xr-x 1 root root 1806600 Jan 31 22:32 ldscope*
-rwxr-xr-x 1 root root 1654608 Jan 31 22:32 libscope.so*
-rw-r--r-- 1 root root    4783 Jan 31 22:32 scope.yml

```

Now you are ready to configure AppScope to instrument any application and output data to any existing tool via simple TCP protocols.

Depending on your use case and preferred way of working, this usually entails editing `scope.yml`, and then either running `ldscope` or setting environment variables while invoking the library.

How the library is loaded depends on the type of executable. A dynamic loader can preload the library (where supported), while AppScope can load static executables. Regardless of how the library is loaded, you get full control of the data source, formats, and transports.

The following examples provide an overview of how to use the library.

Let's go on to [Using the Library](/docs/library-using).

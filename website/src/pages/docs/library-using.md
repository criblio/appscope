---
title: Using the Library
---

## Using the Library (libscope.so)

To use the library for the first time in a given environment, complete this quick procedure:

1. Download AppScope. 

2. Choose a `SCOPE_HOME` directory, i.e., the directory from which AppScope should run in your environment.

3. Extract (`scope extract`) the contents of the AppScope binary into the `SCOPE_HOME` directory.

For example, you could create an AppScope home directory called `assets`:

```
user@myhost:/opt/scope$ mkdir assets
user@myhost:/opt/scope$ bin/linux/scope extract assets
Successfully extracted to /opt/scope/assets.
user@myhost:/opt/scope$ ll assets
total 17008
drwxrwxr-x  2 user user    4096 Oct 26 14:47 .
drwxrwxr-x 18 user user    4096 Oct 26 14:47 ..
-rwxr-xr-x  1 user user 9206704 Oct 26 14:47 ldscope
-rwxr-xr-x  1 user user 8161448 Oct 26 14:47 libscope.so
-rw-r--r--  1 user user   33231 Oct 26 14:47 scope.yml
```

Now you are ready to configure AppScope to instrument any application and output data to any existing tool via simple TCP protocols.

Depending on your use case and preferred way of working, this usually entails editing `scope.yml`, and then setting environment variables while invoking the library.

How the library is loaded depends on the type of executable. A dynamic loader can preload the library (where supported), while AppScope can load static executables. Regardless of how the library is loaded, you get full control of the data source, formats, and transports.

<span id="env-vars"> </span>

### Env Vars and the Config File

To see the full set of library environment variables, run the following command:

```
/opt/scope/assets/ldscope --help | egrep "^[[:space:]]{2}SCOPE_"
```

For the default settings in the sample `scope.yml` configuration file, see [Config File](/docs/config-file), or inspect the most-recent file on [GitHub](https://github.com/criblio/appscope/blob/master/conf/scope.yml).

To see the config file with comments omitted, run the following command:

```
egrep -v '^ *#.*$' scope.yml | sed '/^$/d' >scope-minimal.yml

```

This can help you get a clear idea of exactly how AppScope is configured, assuming you have previously read and understood the comments.

### Using the Library Directly

To use the library directly, you rely on the `LD_PRELOAD` environment variable. 

The following examples provide an overview of this way of working with the library. All the examples call the system-level `ps` command, just to show how the syntax works.

**LD_PRELOAD with a Single Command**

Start with this basic example:

```
LD_PRELOAD=./libscope.so ps -ef
```

This executes the command `ps -ef`. But first, the OS's loader loads the AppScope library, as part of loading and linking the `ps` executable.

Details of the `ps` application's execution are emitted to the configured transport, in the configured format. For configuration details, see [Env Vars and the Config File](#env-vars) above.

**LD_PRELOAD with Verbosity Specified**

```
LD_PRELOAD=./libscope.so SCOPE_METRIC_VERBOSITY=5 ps -ef
```

This again executes the `ps` command using the AppScope library. But it also defines the verbosity for metric extraction as level `5`. (This verbosity setting overrides any config-file setting, as well as the default value.)

**LD_PRELOAD with a Config File**

```
LD_PRELOAD=./libscope.so SCOPE_HOME=/etc/scope ps -ef
```

This again executes the `ps` command using the AppScope library. But it also directs the library to use the config file `/etc/scope/scope.yml`.

**LD_PRELOAD with a TCP Connection**

```
LD_PRELOAD=./libscope.so SCOPE_EVENT_DEST=tcp://localhost:9999 ps -ef
```

This again executes the `ps` command using the AppScope library. But here, we also specify that events (as opposed to metrics) will be sent over a TCP connection to localhost, using port `9999`. (This event destination setting overrides any config-file setting, as well as the default value.)

### Adding AppScope to a `systemd` (boot-time) Service 

In this example, we'll add AppScope to the `httpd` service, described by an `httpd.service` file which contains an `EnvironmentFile=/etc/sysconfig/httpd` entry.

1. Extract the library to a new directory (`/opt/scope` in this example):

```
mkdir /opt/scope && cd /opt/scope
curl -Lo scope https://cdn.cribl.io/dl/scope/\
 $(curl -L https://cdn.cribl.io/dl/scope/latest)/linux/scope && \
 chmod 755 ./scope
./scope extract .
```

The result will be that the system uses `/opt/scope/scope.yml` to configure `libscope.so`.

2. Add an `LD_PRELOAD` environment variable to the `systemd` config file.

In the `httpd.service` file, edit the `/etc/sysconfig/httpd` entry to include the following environment variables:

```
SCOPE_HOME=/opt/scope
LD_PRELOAD=/opt/scope/libscope.so
```

<span id="lambda"> </span>span>

### Deploying the Library in an AWS Lambda Function

You can interpose the `libscope.so` library into an AWS Lambda function as a [Lambda layer](https://docs.aws.amazon.com/lambda/latest/dg/configuration-layers.html), using these steps. By default, Lambda functions use `lib` as their `LD_LIBRARY_PATH`, which makes loading AppScope very easy.

1. Run `scope extract`.

```
mkdir lib
scope extract ./lib
```

2. Modify the `scope.yml` configuration file as appropriate.

3. Compress `libscope.so` into a `.zip` file:

```
tar pvczf lambda_layer.zip lib/
```

4. Create a [Lambda layer](https://docs.aws.amazon.com/lambda/latest/dg/configuration-layers.html#configuration-layers-create), and associate the runtimes you want to use with AppScope in your Lambda functions. 

5. Upload the `lambda_layer.zip` file created in Step 3.

6. Add the custom layer to your Lambda function by selecting the previously created layer and version. 

#### Environment Variables

At a minimum, you must set the `LD_PRELOAD` environment variable in your Lambda configuration:

```
LD_PRELOAD=libscope.so
```

For static executables (like the Go runtime), set `SCOPE_EXEC_PATH` to run `ldscope`:

```
SCOPE_EXEC_PATH=/lib/ldscope
```

You must also tell AppScope where to deliver events. This can be accomplished by setting any one of the following environment variables:

- `SCOPE_CONF_PATH=/opt/scope/assets/scope.yml`

- `SCOPE_EVENT_DEST=tcp://host:port`

- `SCOPE_CRIBL=tcp://host:port`

---
title: Using the Library
---

## Using the Library (libscope.so)

To use the library for the first time in a given environment, complete this quick procedure:

1. [Download](/docs/downloading) AppScope. 
2. Decide on a `SCOPE_HOME` directory, i.e., the [directory from which AppScope should run](/docs/downloading#where-from) in your environment.
3. Set `SCOPE_HOME` to the desired directory.
   ```
   ubuntu@my_hostname:~/someuser/temp$ export SCOPE_HOME=/opt/appscope
   ```
4. Create the `SCOPE_HOME` directory. (Here, `sudo` is required because `opt` is owned by root.)
   ```
   ubuntu@my_hostname:~/someuser/temp$ sudo mkdir $SCOPE_HOME
   ```
5. Extract (`scope extract`) the contents of the AppScope binary into the `SCOPE_HOME` directory.
   ```
   ubuntu@my_hostname:~/someuser/temp$ sudo ./scope extract $SCOPE_HOME
   Successfully extracted to /opt/appscope.
   ```
6. Verify that `SCOPE_HOME` contains the AppScope library (`libscope.so`), the `ldscope` utility, and the config file (`scope.yml`). 
   ```
   ubuntu@my_hostname:~/someuser/temp$ ls -al $SCOPE_HOME
   total 20528
   drwxr-xr-x 2 root root     4096 Jul 11 22:51 .
   drwxr-xr-x 5 root root     4096 Jul 11 22:51 ..
   -rwxr-xr-x 1 root root 11308832 Jul 11 22:51 ldscope
   -rwxr-xr-x 1 root root  9663240 Jul 11 22:51 libscope.so
   -rw-r--r-- 1 root root    35755 Jul 11 22:51 scope.yml
   ubuntu@my_hostname:~/someuser/temp$ 
   ```

Now you are ready to configure AppScope to instrument any application and output data to any existing tool via simple TCP protocols.

Depending on your use case and preferred way of working, this usually entails editing `scope.yml`, and then setting environment variables while invoking the library.

How the library is loaded depends on the type of executable. A dynamic loader can preload the library (where supported), while AppScope can load static executables. Regardless of how the library is loaded, you get full control of the data source, formats, and transports.

<span id="env-vars"> </span>

### Env Vars and the Config File

To see the full set of library environment variables, run the following command:

```
/opt/appscope/ldscope --help | egrep "^[[:space:]]{8}SCOPE_"
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

For more, check out the [Further Examples](/docs/examples-use-cases), which include both CLI and library use cases.

#### `LD_PRELOAD` with a Single Command

Start with this basic example:

```
LD_PRELOAD=./libscope.so ps -ef
```

This executes the command `ps -ef`. But first, the OS's loader loads the AppScope library, as part of loading and linking the `ps` executable.

Details of the `ps` application's execution are emitted to the configured transport, in the configured format. For configuration details, see [Env Vars and the Config File](#env-vars) above.

#### `LD_PRELOAD` with Verbosity Specified

```
LD_PRELOAD=./libscope.so SCOPE_METRIC_VERBOSITY=5 ps -ef
```

This again executes the `ps` command using the AppScope library. But it also defines the [verbosity](/docs/events-and-metrics#metrics) for metric extraction as level `5`. (This verbosity setting overrides any config-file setting, as well as the default value.)

#### `LD_PRELOAD` with a Config File

```
LD_PRELOAD=./libscope.so SCOPE_HOME=/etc/scope ps -ef
```

This again executes the `ps` command using the AppScope library. But it also directs the library to use the config file `/etc/scope/scope.yml`.

#### `LD_PRELOAD` with a TCP Connection

```
LD_PRELOAD=./libscope.so SCOPE_EVENT_DEST=tcp://localhost:9999 SCOPE_CRIBL_ENABLE=false ps -ef
```

This again executes the `ps` command using the AppScope library. But here, we also specify that events (as opposed to metrics) will be sent over a TCP connection to localhost, using port `9999`. (This event destination setting overrides any config-file setting, as well as the default value.)

### Adding AppScope to a `systemd` (boot-time) Service

In this example, we'll add AppScope to the `httpd` service, described by an `httpd.service` file which contains an `EnvironmentFile=/etc/sysconfig/httpd` entry.

1. Extract the library to a new directory (`/opt/appscope` in this example):

```
mkdir /opt/appscope && cd /opt/appscope
curl -Lo scope https://cdn.cribl.io/dl/scope/\
 $(curl -L https://cdn.cribl.io/dl/scope/latest)/linux/scope && \
 chmod 755 ./scope
./scope extract .
```

The result will be that the system uses `/opt/appscope/scope.yml` to configure `libscope.so`.

2. Add an `LD_PRELOAD` environment variable to the `systemd` config file.

In the `httpd.service` file, edit the `/etc/sysconfig/httpd` entry to include the following environment variables:

```
SCOPE_HOME=/opt/appscope
LD_PRELOAD=/opt/appscope/libscope.so
```

<span id="lambda"> </span>

### Deploying the Library in an AWS Lambda Function

You can interpose the `libscope.so` library into an AWS Lambda function as a [Lambda layer](https://docs.aws.amazon.com/lambda/latest/dg/configuration-layers.html). By default, Lambda functions use `lib` as their `LD_LIBRARY_PATH`, which makes loading AppScope easy.

Assuming that you have [created](https://aws.amazon.com/lambda/getting-started/) one or more AWS Lambda functions, all you need to do is add the Lambda layer, then set environment variables for the Lambda function.

#### Adding an AppScope AWS Lambda Layer

1. Start with one of the AWS Lambda Layers for AppScope that Cribl provides. You can obtain the AWS Lambda Layers and their MD5 checksums from the Cribl CDN.
    - `AWS Lambda Layer for x86`: [https://cdn.cribl.io/dl/scope/1.2.0/linux/x86_64/aws-lambda-layer.zip](https://cdn.cribl.io/dl/scope/1.2.0/linux/x86_64/aws-lambda-layer.zip)
    - `AWS Lambda Layer for ARM`: [https://cdn.cribl.io/dl/scope/1.2.0/linux/aarch64/aws-lambda-layer.zip](https://cdn.cribl.io/dl/scope/1.2.0/linux/aarch64/aws-lambda-layer.zip)
    - To obtain the MD5 checksum for either file above, add `.md5` to the file path.
2. Complete the procedure for creating a layer described in the [AWS docs](https://docs.aws.amazon.com/lambda/latest/dg/configuration-layers.html#configuration-layers-create), uploading your AppScope AWS Lambda Layer ZIP file in the **upload your layer code** step, and choosing `x86_64` or `ARM64`, as appropriate, for **Compatible architectures**.
3. After you click **Create**, note the **Version ARN** shown for your newly-created layer.
4. Navigate to **Lambda** > **Layers** > **Add layer**, and in the **Choose a layer** section, select **Specify an ARN**. 
5. Enter your layer's ARN, click **Verify**, and then click **Add**.  

#### Setting the Lambda Function's Environment Variables

The AWS docs [explain](https://docs.aws.amazon.com/lambda/latest/dg/configuration-envvars.html) how to set environmental variables for Lambda functions. You'll need to enter the following AppScope environment variable settings in the AWS UI.

1. `LD_PRELOAD` gets your Lambda function working with AppScope.

    - `LD_PRELOAD=libscope.so`

2. `SCOPE_EXEC_PATH` is required for static executables (like the Go runtime).

    - `SCOPE_EXEC_PATH=/lib/ldscope`

3. To tell AppScope where to deliver events, the required environment variable depends on your desired [Data Routing](/docs/data-routing).

    - For example, `SCOPE_CRIBL_CLOUD` is required for an [AppScope Source](https://docs.cribl.io/stream/sources-appscope) in a Cribl.Cloud-managed instance of Cribl Stream. (Substitute your host and port values for the placeholders.)

    - `SCOPE_CRIBL_CLOUD=tcp://<host>:<port>`

4. Optionally, set additional environment variables as desired. 

    - For example, `SCOPE_CONF_PATH` ensures that your Lambda function uses AppScope with the correct config file. (Edit the path if yours is different.)

    - `SCOPE_CONF_PATH=/opt/appscope/scope.yml`


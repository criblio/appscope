# Use Cases by Component

## Get Started with the CLI

The easiest way to get started with AppScope is to use the CLI. This provides a rich set of capabilities intended to capture data from single applications. Data is captured in the local file system and is managed through the CLI. 

## Use the Loader Interface (Production Environments)

As you get more interested in obtaining details from production applications, explore using the AppScope library apart from the CLI. This allows for full configurability via the `scope.yml` configuration file (default `scope.yml` [here]([https://github.com/criblio/appscope/blob/master/conf/scope.yml](https://github.com/criblio/appscope/blob/master/conf/scope.yml)l)). You can send details over UDP, TCP, and local or remote connections, and you can define specific events and metrics for export. You can use the AppScope loader to start applications that include the library. 

## Use the Library Independently (Production Environments)

You can also load the AppScope library independently, configured by means of a configuration file and/or environment variables. The config file (default: [`scope.yml`](https://github.com/criblio/appscope/blob/master/conf/scope.yml)) can be located where needed – just reference its location using the `LD_PRELOAD` environment variable. Environment variables take precedence over the default configuration, as well as over details in any configuration file.

You use an environment variable to load the library independent of any executable. Whether you use this option or the AppScope loader, you get full control of the data source, formats, and transports. 
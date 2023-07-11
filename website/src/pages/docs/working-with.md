---
title: Working With AppScope
---

## Working With AppScope 

These docs explain two ways to work with AppScope: together with [Cribl Edge](https://docs.cribl.io/edge/), and on its own. Cribl Edge provides a means to manage AppScope at scale. The concluding Reference topics generally apply for both approaches, although the [CLI Reference](/docs/cli-reference) is most relevant for using AppScope on its own or with other open-source tools.

### Using AppScope with Cribl Edge

Here we'll cover working on a Linux host or virtual machine, in a Docker or similar container, or on Kubernetes.

These topics are complementary to Cribl's documentation about the [AppScope Source](https://docs.cribl.io/stream/sources-appscope) and the [AppScope Config Editor](https://docs.cribl.io/stream/4.0/appscope-configs).

AppScope comes bundled with Cribl Edge, so there's no need to discuss how to obtain AppScope. 

### Using AppScope On its Own

Here we'll cover obtaining, and then using AppScope, starting with the CLI.

## Fundamental Concepts {#fundamentals}

To work effectively with AppScope, start with these fundamental points:

* Your overall approach can be either spontaneous or more planned-out.
* You can [control](#config-file-etc) AppScope by means of the config file, environment variables, or a combination of the two.
* The results you get from AppScope will be in the form of [events and metrics](/docs/events-and-metrics). When you scope HTTP applications, you can get payloads, too.

## Scoping By PID and Scoping By Rule {#pid-vs-rule}

Another important distinction to understand when working with AppScope is that between "scope by PID" and "scope by Rule."

* Scope by PID instruments one process on one host or container.
* Scope by Rule instruments one or more processes, not only on one host, but, when using AppScope together with Cribl Edge, an entire Edge Fleet.
    * The principle is that AppScope will instrument whatever processes match a given Rule.

## The Config File, Env Vars, and Flags {#config-file-etc}

AppScope's ease of use stems from its flexible set of controls:

* AppScope's configuration file, `scope.yml`, can be invoked from either the CLI or the library.

* The AppScope library provides an extensive set of environment variables, which control settings like metric verbosity and event destinations. Environment variables override config file settings.

Check out the [CLI](/docs/cli-using) and [library](/docs/library-using) pages to see how it's done.

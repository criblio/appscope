---
title: Working With AppScope
---

# Working With AppScope

You can operate AppScope in two "modes:" 

- In Edge mode, you "drive" AppScope from Cribl Edge.
- In Standalone mode, you "drive" AppScope from the command line. Optionally, you can use AppScope send data to Cribl Edge, Cribl Stream, or any other suitable application or platform.

The way these docs are organized follows from two-modes concept:
- When using AppScope with Cribl Edge, there are sections about working on a Linux host or virtual machine, in a Docker or similar container, or on Kubernetes.
    - These sections are complementary to Cribl's documentation about the [AppScope Source](https://docs.cribl.io/stream/sources-appscope) and the [AppScope Config Editor](https://docs.cribl.io/stream/4.0/appscope-configs).
    - AppScope comes bundled with Cribl Edge, so there's no need to discuss how to obtain AppScope. 
- When using AppScope on its own, there are sections for obtaining AppScope, and then using AppScope, starting with the CLI.
- The Reference sections apply to both modes (although the [CLI Reference](/docs/cli-reference) is most relevant for using AppScope on its own or with other open-source tools).

There are three main things to know to work effectively with AppScope:

* Your overall approach can be either spontaneous or more planned-out.
* You can [control](#config-file-etc) AppScope by means of the config file, environment variables, or a combination of the two.
* The results you get from AppScope will be in the form of [events and metrics](/docs/events-and-metrics). When you scope HTTP applications, you can get payloads, too.

## The Config File, Env Vars, and Flags {#config-file-etc}

AppScope's ease of use stems from its flexible set of controls:

* AppScope's configuration file, `scope.yml`, can be invoked from either the CLI or the library.

* The AppScope library provides an extensive set of environment variables, which control settings like metric verbosity and event destinations. Environment variables override config file settings.

Check out the [CLI](/docs/cli-using) and [library](/docs/library-using) pages to see how it's done.

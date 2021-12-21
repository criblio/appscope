---
title: Downloading AppScope
---

## Downloading AppScope

Download AppScope, then explore its CLI. Getting started is that easy.

You can download AppScope as a binary to run from a [suitable location](#where-from) in your Linux OS, or as a container.

### Download as Binary

Review the AppScope [Requirements](/docs/requirements) to ensure that you’re completing these steps on a supported system. 

Then, from [Cribl's download page](https://cribl.io/download/#tab-1), download the binary and follow the instructions there.

Or, you can use these CLI commands to directly download the binary and make it executable:

```
LATEST=$(curl -Ls https://cdn.cribl.io/dl/scope/latest)
curl -Lo /opt/scope https://cdn.cribl.io/dl/scope/$LATEST/linux/$(uname -m)/scope
curl -Ls https://cdn.cribl.io/dl/scope/$LATEST/linux/$(uname -m)/scope.md5 | md5sum -c 
chmod +x /opt/scope
```

### Download as Container

From [Cribl's download page](https://cribl.io/download/#tab-1) you can also download AppScope's most recently tagged Ubuntu container.

This will link you to the [AppScope repo on Docker Hub](https://hub.docker.com/r/cribl/scope), which provides instructions and access to earlier builds.

Each container provides the AppScope binary on Ubuntu 20.04.

### Verify AppScope

Run some scope commands:

```
scope run ps -ef
scope events
scope metrics
```

That's it!

Now you are ready to explore the CLI:

- Run `scope --help` or `scope -h` to view CLI options.
- Try some basic CLI commands in [Using the CLI](/docs/cli-using).
- See the complete [CLI Reference](/docs/cli-reference).

<span id="where-from"> </span>

### What's the Best Location to Run AppScope From?

Where AppScope lives on your system is up to you. To decide wisely, first consider which of the [two ways of working](/docs/working-with) with AppScope fit your situation: ad hoc or planned-out.

For ad hoc exploration and investigation, one choice that follows [standard practice](https://en.wikipedia.org/wiki/Filesystem_Hierarchy_Standard) for an "add-on software package" is to locate AppScope in `/opt`. Cribl does not recommend running AppScope from any user's home directory, because that leads to file ownership and permissions problems if additional users try to run AppScope.

By default, the AppScope CLI will output data to the local filesystem where it's running. While this can work fine for "ad hoc" investigations, it can cause problems for longer-duration scenarios like application monitoring, where the local filesystem may not have room for the data being written. 

Also consider what level of logging you want from AppScope, and where those logs should go. In planned-out scenarios, where AppScope runs persistently (including as a service), configuring logging judiciously can prevent AppScope from using up too much disk space. 

To customize AppScope's data-writing and logging, edit the [`scope.yml`](/docs/config-file) config file.

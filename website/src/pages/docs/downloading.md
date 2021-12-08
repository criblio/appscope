---
title: Downloading AppScope
---

## Downloading AppScope

Download AppScope, then explore its CLI. Getting started is that easy.

You can download AppScope as a binary to run on your Linux OS, or as a container.

### Download as Binary

First, see [Requirements](/docs/requirements) to ensure that you’re completing these steps on a supported system. 

Next, use these commands to download the CLI binary and make it executable:

```
LATEST=$(curl -Ls https://cdn.cribl.io/dl/scope/latest)
curl -Lo scope https://cdn.cribl.io/dl/scope/$LATEST/linux/$(uname -m)/scope
curl -Ls https://cdn.cribl.io/dl/scope/$LATEST/linux/$(uname -m)/scope.md5 | md5sum -c 
chmod +x scope
```

### Download as Container

Visit the AppScope repo on Docker Hub to download and run the most recently tagged container:

https://hub.docker.com/r/cribl/scope

The container provides the AppScope binary on Ubuntu 20.04.


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

### What's the Best Location to Run AppScope From?

Where AppScope lives on your system is up to you. To decide wisely, first consider which of the [two ways of working](/docs/working-with) with AppScope fit your situation: ad hoc or planned-out.

For ad hoc exploration and investigation, one choice that follows [standard practice](https://en.wikipedia.org/wiki/Filesystem_Hierarchy_Standard) for an "add-on software package" is to locate AppScope in `/opt`. Running AppScope from a user's home directory is not recommended.

You should consider what level of logging you want from AppScope, and where those logs should go, as configured in [`scope.yml`](/docs/config-file). In planned-out scenarios, where AppScope runs persistently (including as a service), these choices can prevent AppScope from using up too much disk space. 

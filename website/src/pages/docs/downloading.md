---
title: Downloading AppScope
---

## Downloading AppScope

Download AppScope, then explore its CLI. Getting started is that easy.

You can download AppScope as a binary to run on your Linux OS, or as a container.

### Download as Binary

First, see [Requirements](/docs/requirements) to ensure that you’re completing these steps on a supported system. 

Next, you can download the most-recent binary from [Cribl's download page](https://cribl.io/download/#tab-1) and follow the instructions there.

Or, you can use these CLI commands to directly download the binary and make it executable:

```
LATEST=$(curl -Ls https://cdn.cribl.io/dl/scope/latest)
curl -Lo scope https://cdn.cribl.io/dl/scope/$LATEST/linux/$(uname -m)/scope
curl -Ls https://cdn.cribl.io/dl/scope/$LATEST/linux/$(uname -m)/scope.md5 | md5sum -c 
chmod +x scope
```

### Download as Container

You can also download AppScope's' most recently tagged Ubuntu container from [Cribl's download page](https://cribl.io/download/#tab-1).

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

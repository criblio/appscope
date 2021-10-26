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

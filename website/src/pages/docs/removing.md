---
title: Removing
---

## Removing

You can remove AppScope by simply deleting the binary (along with the rest of the contents of the `SCOPE_HOME` directory):

```
rm -rf /opt/appscope/
```

…and the associated history directory:

```
cd ~
rm -rf .scope/
```
</br>

Currently scope’d applications will continue to run. To remove the AppScope library from a running process, you will need to restart that process.

### Muting Without Removing

If you need AppScope to temporarily stop producing output, you do not need to remove AppScope: you can just use a custom config to [mute](data-routing#muting) all events and metrics.

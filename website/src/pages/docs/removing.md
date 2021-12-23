---
title: Removing
---

## Removing

You can remove AppScope by simply deleting the binary (along with the rest of the contents of the `scope` directory):

```
rm -rf /opt/scope/
```

…and the associated history directory:

```
cd ~
rm -rf .scope/
```
</br>

Currently scope’d applications will continue to run. To remove the AppScope library from a running process, you will need to restart that process.

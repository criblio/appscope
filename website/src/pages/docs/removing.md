---
title: Removing
---

## Removing

You can remove AppScope by simply deleting the binary, along with the rest of the contents of the `SCOPE_HOME` directory. For example, if your `SCOPE_HOME` directory is `/opt/appscope/`:

```
rm -rf /opt/appscope/
```

Then delete the associated history directory:

```
cd ~
rm -rf .scope/
```
</br>

Currently scope’d applications will continue to run. To remove the AppScope library from a running process, use `scope detach <process_ID>`.

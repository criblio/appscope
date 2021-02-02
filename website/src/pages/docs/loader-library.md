---
title: Extracting the Library
---

## The Loader, Library and .yml Files
---
As mentioned in [How AppScope Works](/docs/how-works), the single binary contains a **Loader** and a **Library** component that can be used independently of the CLI. In order to use them, they must first be extracted using the `extract` sub-command. This will output `ldscope`, `libscope.so`, `scope.yml` and `scope_protocol.yml` in a directory of choice. These files can be later configured to instrument any application and output the data to any existing tool using simple TCP protocols. 



### Using extract 

The syntax is as follows: `scope extract <dir_to_extract>`. E.g., 

```
# mkdir assets && scope extract assets
Successfully extracted to assets.


# ll assets/
total 3404
drwxr-xr-x 2 root root    4096 Jan 31 22:32 ./
drwxr-xr-x 1 root root    4096 Jan 31 22:32 ../
-rwxr-xr-x 1 root root 1806600 Jan 31 22:32 ldscope*
-rwxr-xr-x 1 root root 1654608 Jan 31 22:32 libscope.so*
-rw-r--r-- 1 root root    4783 Jan 31 22:32 scope.yml
-rw-r--r-- 1 root root     181 Jan 31 22:32 scope_protocol.yml

```

### Next Steps

Let's go to [Using the Library](/docs/library-using).
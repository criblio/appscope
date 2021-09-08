
To make the unix tests work, we need to enable file-backed unix sockets in src/transport.c:transportCreateUnix

```
    strncpy(&trans->local.addr.sun_path[1], path, pathlen);
```
change to:
```
    strncpy(&trans->local.addr.sun_path[0], path, pathlen);
```

To include the tests in our integration workflow we need to uncomment the fluentbit section in testContainers/docker-compose.yml

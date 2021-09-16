
Add integration tests to discover compatibility between appscope and fluentbit.

To include the tests in our integration workflow we need to uncomment the fluentbit section in testContainers/docker-compose.yml

# Build

```
docker build -t test .
```

# Run

```
docker run -v "/var/run/docker.sock:/var/run/docker.sock" test
```

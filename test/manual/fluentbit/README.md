
Manual tests to discover compatibility between appscope and fluentbit, using our appscope input and output plugins.

The docker-compose file defines a service 'fluentbit' that runs the tests in 'scope-test'. 'scope-test' is able to use the same docker-compose file to bring up dependencies in parallel containers.

Essentially, all container definitions are flat-packed into one central docker-compose file containing all testing dependencies.

This approach is a trial at docker-in-docker, following discussions around viability/usefulness of running dependent services in their own containers vs all services in one container.

# Build

```
docker-compose build fluentbit
```

# Run

```
docker-compose run fluentbit
```

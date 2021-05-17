# AppScope Integration Tests

We run various integration tests here to exercise AppScope in a number of
real-world scenarios. Each test has a Docker container image that provides 
the runtime environment. They are self-sufficent and self-executable. The
general strategy is to run set of tests/challenges against target application
or syscall in order to determine that target is working correctly normally.
Then, to run the same set of tests against the same target, but wrapped in
AppScope (_scoped_) and compare results with raw execution.

## Configs

Each test has a corresponding _service_ in `docker-compose.yml`. They have a
corresponding subdirectory with their `Dockerfile` and `docker-entrypoint.sh`
along with any additional content that needs to be included in the image.

All of the tests use `/docker-entrypoint.sh` as their `ENTRYPOINT` and `test`
as their default `CMD`. We can `docker run test /bin/sh` to get a shell into
the containers to poke around.

We mount the locally built AppScope binaries into each container when it's run
so they are always in the same place. If you've not already built those
binaries, running these tests will fail when they try to mount them. They get
mounted into the container as follows.

- `/usr/bin/scope`
- `/usr/bin/ldscope`
- `/usr/lib/libscope.so`

To speed up using these tests, we keep the built container image at Docker Hub
as `criblci/scope-(test)-it:latest`. We're not tagging separate versions for
branches at this point. We'll revisit that later if we start stepping on each
other.

## Execution

The `Makefile` provides the automation. Run `make help` for details as well as
a list of all the tests. 

```shell
$ make help
AppScope Integration Test Runner
  `make help` - show this info
  `make all` - run all tests
  `make build` - build all test images
  `make push` - push all test images
  `make (test)` - run a single test
  `make (test)-shell` - run a shell in the test's container
  `make (test)-build` - build the test's image
Tests: bash cribl detect-proto elastic gogen go-1.2 go-1.3 go-1.4 go-1.5 go-1.6 go-1.7 go-1.8 go-1.9 go-1.10 go-1.11 go-1.12 go-1.13 go-1.14 go-1.15 go-1.16 java6 java7 java8 java9 java10 java11 java12 java13 java14 kafka nginx oracle-java6 oracle-java7 oracle-java8 oracle-java9 oracle-java10 oracle-java11 oracle-java12 oracle-java13 oracle-java14 splunk syscalls tls
$
```

Developers can run tests locally pretty easily with this setup. We use it for
CI/CD logic too.

## Tests

As mentioned before, start with the `docker-compose.yml` file and each entry's
`Dockerfile` and `docker-entrypoint.sh` files for details on what each test is
doing. Some notes of a few specific tests are below.

### `cribl`

Test the standalone version of Cribl LogStream. 

### `splunk`

Tests Splunk Enterprise. Splunk instance has Cribl LogStream as Splunk
Application, but LogStream capabilities are not tested or used.

### `syscalls`

Tests Linux syscalls supported by Scope using executable syscall unit tests.
Some unit tests provided by [LTP Project][LTP] 
the others are custom C
test located in [syscalls/altp folder](syscalls/altp)

### `nginx`

Tests Nginx web server using Apache Benchmark tool.


## `test-runner` Framework

Some of the tests use a framework in the `./test-runner/` directory. In
addition to logging to the console, they dump logs to `/opt/test-runner/logs`
which we mount here to `./logs`. 

[LTP]: https://github.com/linux-test-project/ltp

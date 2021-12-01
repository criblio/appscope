# AppScope Integration Tests

We run integration tests here to exercise AppScope in real-world scenarios.
Each test has a Docker container image that provides the runtime environment.
They are self-sufficent and self-executable. The general strategy is to run set
of tests/challenges against target application or syscall in order to determine
that target is working correctly normally. Then, to run the same set of tests
against the same target, but wrapped in AppScope (_scoped_) and compare results
with raw execution.

## Configs

Each test has a corresponding _service_ in `docker-compose.yml`. They have a
corresponding subdirectory with their `Dockerfile` along with any additional
content that needs to be included in the image. When run, we top-level working
directory (i.e `../../` from here) is mounted at `/opt/appscope` so the built
binaries (i.e. `scope`, `ldscope`, and `libscope.so`) are accessible in the
container. Since they're in subdirectories in the mount, they can be rebuilt
and the container will have access to the updated versions without having to
restart - handy when debugging.

We have moved some of the test service definitions out of `docker-compose.yml`
into separate arch-specific configs; `docker-compose.$(uname -m).yml`. This
was done because the base images for some of them are not available on ARM.

## Images

We're trying to make all of the images as consistent as we can so we can hit
the ground running when we need to dig into a failure. Here's what we expect
from each one.

* As mentioned above, the top working directory is mounted at `/opt/appscope`.
  The Dockerfiles setup softlinks in `/usr/local/scope/bin` and `.../lib` that
  point to the built binaries in `/opt/appscope`. We add the `bin/` folder to
  the path too so `scope` and `ldscope` are accessible anywhere.

* ENTRYPOINT is always `/docker-entrypoint.sh` and CMD defaults to `test`. If
  an alternate CMD is not specified when the container is run, this causes the
  `/usr/local/scope/scope-test` script to be run. It too is in the PATH. Each
  test installs their own version of the `scope-test` command.

* We _cache_ the container images in Docker Hub repositories for each test
  named `scopeci/scope-${TEST}-it`. The Makefile will pull them down and use
  them to speed up building the local versions. We use `make push` to push
  them back up to Docker Hub. 

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
  `make (test)-exec` - run a shell in the existing test's container
  `make (test)-build` - build the test's image
Tests: alpine attach-glibc attach-musl bash cli console detect_proto elastic glibc gogen go_2 go_3 go_4 go_5 go_6 go_7 go_8 go_9 go_10 go_11 go_12 go_13 go_14 go_15 go_16 go_17 http java7 java8 java9 java10 java11 java12 java13 java14 kafka logstream nginx oracle-java7 oracle-java8 oracle-java9 oracle-java10 oracle-java11 musl oracle-java12 oracle-java13 oracle-java14 service-initd service-systemd splunk syscalls syscalls-alpine tls transport
```

Developers can run tests locally pretty easily with this setup. We use it for
CI/CD logic too.

Set `START=${TEST}` to the name of a test when running `make all` to skip the
tests ahead of it. Useful when one fails and you want to pickup where you left
off; i.e. `make all START=logstream` skips the test before `logstream`.

## Tests

As mentioned before, start with the `docker-compose.yml` file and each entry's
`Dockerfile` and `docker-entrypoint.sh` files for details on what each test is
doing. Some notes of a few specific tests are below.

### `cribl`

Test the standalone version of Cribl LogStream. 

### `splunk`

Tests Splunk Enterprise. Splunk instance has Cribl LogStream as Splunk
Application, but LogStream capabilities are not tested or used.

### `syscalls` and `syscalls-alpine`

Tests Linux syscalls supported by Scope using executable syscall unit tests.
Some unit tests provided by [LTP Project][LTP]; others are custom C tests
located in [syscalls/altp folder](syscalls/altp).

This test is explicitly using `LD_PRELOAD` while other tests are using a mix of
`scope` and `ldscope` and `LD_PRELOAD`. We have been removing use of
`LD_PRELOAD` from some other tests to speed them up but we'll keep it here to
make sure that startup scheme is getting wrung out.

### `nginx`

Tests Nginx web server using Apache Benchmark tool.


## `test-runner` Framework

Some of the tests use a framework in the `./test-runner/` directory. In
addition to logging to the console, they dump logs to `/opt/test-runner/logs`
which we mount here to `./logs`. 

[LTP]: https://github.com/linux-test-project/ltp

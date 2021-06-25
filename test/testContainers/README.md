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

We have an additional config named `docker-compose.privileged.yml` that sets
the privileged option in each container when run with `make ${TEST}-shell` so
we can use `gdb` and other restricted tools.

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
  named `criblci/scope-${TEST}-it`. The Makefile will pull them down and use
  them to speed up building the local versions. We used `make push` to push
  them back up to Docker Hub. 

  We use `README.${TEST}.md` files if found in this directory to update the
  "Overview" tab for the repository.

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
Tests: alpine bash cribl detect-proto elastic gogen go-1.2 go-1.3 go-1.4 go-1.5 go-1.6 go-1.7 go-1.8 go-1.9 go-1.10 go-1.11 go-1.12 go-1.13 go-1.14 go-1.15 go-1.16 java6 java7 java8 java9 java10 java11 java12 java13 java14 kafka nginx oracle-java6 oracle-java7 oracle-java8 oracle-java9 oracle-java10 oracle-java11 oracle-java12 oracle-java13 oracle-java14 splunk syscalls tls
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

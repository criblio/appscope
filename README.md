# Cribl AppScope

AppScope is an open source, runtime-agnostic instrumentation utility for any Linux command or application. It helps users explore, understand, and gain visibility with **no code modification**.

AppScope provides the fine-grained observability of a proxy/service mesh, without the latency of a sidecar. It emits APM-like metric and event data, in open formats, to existing log and metric tools.

It’s like strace meets tcpdump – but with consumable output for events like file access, DNS, and network activity, and StatsD-style metrics for applications. AppScope can also look inside encrypted payloads, offering WAF-like visibility without proxying traffic.

This `README` gets you up and scoping ASAP. Before you begin, make sure that your environment meets AppScope [requirements](https://appscope.dev/docs/requirements). 

To go further:

- Complete the [Cribl AppScope Fundamentals sandbox](https://sandbox.cribl.io/course/appscope), a tutorial that takes about 30 minutes.
- See the [AppScope documentation](https://appscope.dev/docs/) on the Cribl website to learn about AppScope concepts and operation. These docs includes release notes, how-to's, and examples, and are built from the [website/](./website/) content in this repo.
- See the [docs/](./docs/) directory in this repo if you're thinking about contributing to this project, or if you just want to understand the internal logic of AppScope.
- Join the [Cribl Community](https://cribl.io/community/) on Slack. The `#appscope` channel is where you'll find developers who contribute to this project.
- Please submit any feature requests and defect reports at <https://github.com/criblio/appscope>.

## Get AppScope

You can obtain AppScope in any of three ways:

- Get the AppScope Docker container image from Docker Hub, and then [build](#dock) and run AppScope in a container.
- Check out the code from this project, then [build](#nix) and run AppScope in your Linux environment.
- Get the binaries from the Cribl CDN, and run them in your Linux environment.


A fresh Docker container image and binaries are automatically created for each new release tag in this project.

Our goal is for AppScope to be a "build-once, run-anywhere" executable. For that reason: 

- AppScope built from this project gives you Ubuntu **18.04**. We build on this somewhat older distro so that when test on other distros, we can take advantage of backward compatibility.
- AppScope built from from the container image gives you the current version of Ubuntu by default — as of this writing, **20.04**.

### <span id="dock"> Build with Docker </span>

Assuming Docker is installed on your local machine, use the `docker/builder/Dockerfile` and the `make docker-build` target.

(If Docker is **not** installed on your machine, you must install the dependencies locally.)

```
make docker-build
```

This builds an `appscope-builder` image containing Ubuntu with required dependencies installed. The target runs the image, mounting the local directory into the container, then builds the project. The binary should appear in `bin/linux/scope`.

By default, the container image uses the current versions of Ubuntu and Go. If you need to override these, use the `BUILD_ARGS` environment variable to pass `IMAGE` and `GOLANG` as extra arguments to `docker build`.  

The example below forces the builder to use Go 1.16 and Ubuntu 21.04.

```
make docker-build BUILD_ARGS="--build-arg GOLANG=golang-1.16 --build-arg IMAGE=ubuntu:21.04"
```

See the `Makefile` for the `docker run` command we use. To fiddle around in the builder, get a shell by running `docker run` without the `--entrypoint` and `-c` arguments.

### <span id="nix"> Build on Linux </span>

Following the principle of "build-once, run-anywhere," we only support building on one environment — Ubuntu 18.04. If you are running something other than Ubuntu 18.04, consider building with Docker.

Check out the code, install build tools, build, and run tests:

```
git clone https://github.com/criblio/appscope.git
cd appscope
./scope_env.sh build
```

`./scope_env.sh build` gets a brand new, complete environment up and running. In an existing environment, you can run it to rebuild, and to rerun tests. You can also run individual parts of `./scope_env.sh build` from the `appscope` directory:

```
./install_build_tools.sh
make all
make test
```

To clean out files that were added manually or by build commands:

```
git clean -f -d
```

## Start Scoping

Try these simple AppScope CLI commands to verify that AppScope is up and running. 

### Scope a Linux Command

Attach AppScope to the new process that starts when you run a command. 

```
scope ps -ef
```

Try substituting `top` or `curl https://google.com` for `ps -ef`.

In each case, AppScope captures the output of the command you scope.



### Scope a Running Process

AppScope can attach to a process that *is already running*.

Run `ps -ef` and find the process ID (PID) of a command or program you would like to scope.

Attach to and scope the running process:

```
scope attach PID
```

### Keep Going!

Now, in whatever order suits your needs: 

- Check out the CLI [in more depth](https://appscope.dev/docs/quick-start-guide/).
- Get an [overview](https://appscope.dev/docs/how-works/) of AppScope beyond the CLI.
- Look into the [logic]() that governs AppScope components — this is for developers considering contributing to the project.
- Discover what people are [doing](https://appscope.dev/docs/what-do-with-scope) with AppScope.
- Review advanced [examples](https://appscope.dev/docs/examples-use-cases).
- See what happens when you [connect AppScope to Cribl LogStream](https://appscope.dev/docs/logstream-integration).

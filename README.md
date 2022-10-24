[![Build & Test](https://github.com/criblio/appscope/actions/workflows/build.yml/badge.svg)](https://github.com/criblio/appscope/actions/workflows/build.yml)

# AppScope

AppScope is an open source, runtime-agnostic instrumentation utility for any Linux command or application. It helps users explore, understand, and gain visibility with **no code modification**.

AppScope provides the fine-grained observability of a proxy/service mesh, without the latency of a sidecar. It emits APM-like metric and event data, in open formats, to existing log and metric tools.

It’s like [strace](https://github.com/strace/strace) meets [tcpdump](https://www.tcpdump.org/) – but with consumable output for events like file access, DNS, and network activity, and StatsD-style metrics for applications. AppScope can also look inside encrypted payloads, offering WAF-like visibility without proxying traffic.

<br />
<br />

```mermaid
graph LR
    A[Application] --> B[libscope]
    A[Application]--> C[libgnutls]
    A[Application]--> D[libc]
    C --> D
    B --> D
    B --> C
    D --> I[Kernel]
    B --> E[In-memory Queue]
    E -.-> F[Reporting Thread]
    F --> G[Network Destination]
    F --> H[File System Destination]
    style B fill:#f3ffec,stroke:#89db70
    style E fill:#fafafa,stroke:#a6a6a6
    style F fill:#fafafa,stroke:#a6a6a6
    style G fill:#fafafa,stroke:#a6a6a6
    style H fill:#fafafa,stroke:#a6a6a6
```

## Get Started

Before you begin, make sure that your environment meets AppScope [requirements](https://appscope.dev/docs/requirements).

Next, you can obtain AppScope three ways:

- Get the [container image](https://hub.docker.com/r/cribl/scope) from Docker Hub and run AppScope in it.
- Check out the code from this project, then [build](#build) and run AppScope in your Linux environment.
- Get the binaries from the [CDN](docs/RELEASE.md#cdn), and run them in your Linux environment.  

The container image at Docker Hub and binaries at the CDN are updated for each new release of this project.

Once you have downloaded and/or built AppScope, try some simple commands to verify that AppScope is available:

- Scope a new process like `scope ps -ef`. Try substituting `top` or `curl https://google.com` for `ps -ef`.  Run `scope events` to see the results.

- Attach to a process that *is already running*. Run `ps -ef` and find the process ID (PID) of a command or program you would like to scope. Attach to and scope the running process with `scope attach PID`. Run `scope dash` to watch events live.

See the [website docs](https://appscope.dev/docs/overview) for the full story of how to use AppScope.

## Build

AppScope is not built or distributed like most traditional Linux software.

- Insofar as possible, we want AppScope binaries to be  **Build Once, Run Anywhere**. To approach this goal, we build with a version of glibc that is (1) recent enough that the resulting binary contains references to versions of functions in the glibc library *that are still supported in the latest glibc*, yet (2) old enough that the binaries can run on a wide range of Linux platforms without having to rebuild locally.
.
- We don't build OS installation packages like DEBs or RPMs. This way, when you want to investigate a running system or build a custom container image, you can simply drop AppScope in and use it.

Pull a copy of the code with:

```text
git clone https://github.com/criblio/appscope.git
cd appscope
```

If you are on Ubuntu 18.04, install the build dependencies with:

```text
./install_build_tools.sh
```

Then, build and test the code with:

```text
make all test
```

If you are on a platform other than Ubuntu 18.04, or prefer not to install the dependencies, ensure that [Docker], [BuildX], and `make` are installed, then build in a container with:

```text
make build
```

Either way, the resulting binaries will be in `lib/linux/$(uname -m)/libscope.so`, `bin/linux/$(uname -m)/scope`, and `bin/linux/$(uname -m)/ldscope`.

We support building `x86_64` (amd64) or `aarch64` (arm64/v8) binaries by adding `ARCH=x86_64` or `ARCH=aarch64` to the `make build` command. See the [BUILD](docs/BUILD.md) doc for details.

## Keep Going

On the [AppScope Website](https://appscope.dev/):

- Check out the CLI [in more depth](https://appscope.dev/docs/cli-using).
- Get an [overview](https://appscope.dev/docs/how-works/) of AppScope beyond the CLI.
- Discover what people are [doing](https://appscope.dev/docs/what-do-with-scope) with AppScope.
- Review advanced [examples](https://appscope.dev/docs/examples-use-cases).
- View the [Changelog](https://appscope.dev/docs/changelog) and [Known Issues](https://appscope.dev/docs/known-issues).
- See what happens when you [connect AppScope to Cribl Stream or Cribl Edge](https://appscope.dev/docs/cribl-integration).

The content on that site is built from the [website/](website/) directory in this project.

Elsewhere, you can:

- Complete the [AppScope Fundamentals sandbox](https://sandbox.cribl.io/course/appscope), a tutorial that takes about 30 minutes.
- See our Design Docs in the [docs/](./docs/) directory in this repo if you're thinking about contributing to this project, or if you just want to understand the internal logic of AppScope.
- Join the [Cribl Community](https://cribl.io/community/) on Slack. The `#appscope` channel is where you'll find developers who contribute to this project.
 
Please submit any feature requests and defect reports at <https://github.com/criblio/appscope>.

[Docker]: https://docs.docker.com/engine/install/
[BuildX]: https://docs.docker.com/buildx/working-with-buildx/

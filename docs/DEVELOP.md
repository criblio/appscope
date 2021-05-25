# Cribl AppScope - Developer Notes

## Tools

Most of us are on MacOS laptops and runs thing locally either in VMs or
containers. Some of us use Ubuntu laptops instead. We also use EC2 instances
at AWS to work in. Whatever works best for you.

The code is in the GitHub repository at <https://github.com/criblio/appscope>.
It's open-source and public so anyone can pull and build it or submit issues.
You need to be added to the project as a member to be able to commit changes.
Issues and PRs from non-members are welcome.

You'll need Git to pull the code and then GCC, Make, Automake, Autoconf, etc.
to build it. Run the `bootstrap` script to initialize the build system and it
will prompt you to install whatever is missing. See [BUILD.md](./BUILD.md) for
background on the build system.

## Docker Build

The build system provides the standard `make`, `make check`, `make clean`, etc.
targets. One of the [Design Concepts](#design-concepts) is to build the
binaries on older hosts so the resulting glibc dependencies don't require newer
versions. When the local machine is too new, we need a way to build on
something older. We have a `docker-build` target in the Makefile for this. It
builds a container image using [`docker/builder/`](../docker/builder/) then
runs that container with the current directory mounted at `/root/appscope` and
builds. The `docker-run` target uses the same target but provides a shell
instead of running the build.

> It's annoying but after building in a container like this, the generated
content is owned by root so, `sudo chown -R $USER:$USER .` 

## Design Concepts

These are some aspects of the design that weren't immediately obvious to the
new guy.

* AppScope, aka "scope", is a tool much like `strace` that provides a way to
  watch and log the activity of a binary program. It uses a few different
  mechanisms to "interpose" the operations we want to track. We end up causing
  the "scoped" program to call our versions of `read()`, `write()`, `open()`, 
  `close()`, etc. In our versions, we pass through to the original version of
  each routine, log the action, and return the results back to the caller.

* Unlike strace, scope isn't limited to system calls. It interposes a number of
  higher level functions in common libraries (i.e. libc, libtls, libresolv,
  etc.) as well as the Go and Java runtimes. We can log DNS lookups,
  unencrypted reads and writes, etc. without an APM layer being compiled in.

* One of the goals of the project is to produce a single binary that can be
  copied to a machine and used without having to install it or add any other
  dependencies. To that end, the `ldscope` binary has an embedded binary copy
  of `libscope.so` in it that gets dumped out to shared memory and dlopen'ed.
  Similarly, `ldscope` and `libscope.so` are embedded as binary data in the
  `scope` executable. Copy `scope` to a host and you have all you need to use
  it.

* Visible symbols in `libscope.so` need to be kept at a minimum since it is
  `LD_PRELOAD`'ed. Only routines we intend to "interpose" should be visible.
  See [this][1] for how `LD_PRELOAD` is being used.

* We build on machines with older versions of dependencies like libc so the
  resulting binaries are portable to wide array of platforms. To that end,
  newer versions of glibc like the one on Ubuntu 21.04 is too new. Resulting
  binaries fail on older machines. The `make docker-build` feature helps here.

* The `cribl/cribl` container image referenced in the DockerFiles under
  `test/testContainers` is Cribl LogStream. The logic that built it is in the
  internal `/cribl/cribl/src/master/apps/docker-standalone/` project folder;
  not here obviously. The stock ENTRYPOINT is a custom script that runs 
  `cribl server` and tails the log if the CMD is `cribl`. Otherwise, it skips
  running the server and just runs the given CMD.

* Three approaches being used to "interpose" functions for logging. 

  * Preload `libscope.so` using `DL_PRELOAD` to interpose the functions and use
    a constructor in the shared library to setup things and start the thread to
    handle interfaces to LogStream, local configs, etc. This works well for
    dynamic executables. Simplest and cleanest approach.

  * Using "trampoline" functions via FuncHook that munges the in-memory copy of
    the scoped program's code to jump through the trampoline and our handlers
    then back. Used for static binaries.

  * Using GOT (global object table) hooks.

  > It was explained but now I'm not certain about when the last two are used.

* We are required execute the scoped program even when something is wrong with
  the config or other issues early in the startup process. We may be in a
  situation where we're scoping something that cannot not run. This means we
  can end up with silent failures and potentially irritating the user.

* The term "transport" in the code refers an abstraction of the outputs from
  scope; i.e. a file or TCP, UDP or TLS socket.

* cfg.h|c defines the runtime config structure while cfgutils.h|c defines the
  operations to serialize and deserialize the config structure to and from
  JSON and YAML.

* Scope sends events and metrics to configured and enabled destictions that are
  either files, TCP sockets, UDP sockets, or TLS/TCP sockets via the
  `--eventdest` and `--metricdest` options or their corresponding configs. It
  can be configured to send both to Cribl LogStream with the `--cribldest`
  option and configs. The connection to LogStream will support inbound control
  commands as well as outbound log data.

* We used to build for MacOS but that support has not been actively maintained
  for some time. We can look to re-enable that in the future.

[1]: http://www.goldsborough.me/c/low-level/kernel/2016/08/29/16-48-53-the_-ld_preload-_trick/

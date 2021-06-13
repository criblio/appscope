# Cribl AppScope - Startup Logic

AppScope needs to detect and handle a number of different scenarios on startup.
This document is an attempt to capture this logic.

There are three ways to start a new _scoped_ process:

* Using the [CLI](#cli); i.e. `scope run foo`
* Using the [Static Launcher](#static-launcher); i.e. `ldscope foo`
* Using [`LD_PRELOAD`](#ld_preload); i.e. `LD_PRELOAD=libscope.so foo`

There are two ways to _scope_ and existing process:

* Using the [CLI](#cli); i.e. `scope attach PID`
* Using the [Static Launcher](#static-launcher); i.e. `ldscope --attach PID`

We'll cover how each approach works in the sections that follow.

## Library Directory

We need a "Library Directory" where we can put things needed for various
startup scenarios. We need a separate one for each version of AppScope too.

`${libdirbase}` is the base directory where we will create the Library
Directory. It defaults to `/tmp` and may be overriden with command-line
options to the [CLI](#cli) or [Static Launcher](#static-launcher). Startup fails
if `${libdirbase}` doesn't exist or we can't create the Library Directory under
it.

`${libdir}` is our Library Directory; `${libdirbase}/libscope-${version}`.

## Static Launcher

We refer to `ldscope` as the "Static Launcher". It's responsible ultimately for
launcing the [Dynamic Launcher](#dynamic-launcher). Since we introduced musl libc
support, `ldscope` is a static executable and can run without libc. Prior to
musl support being added, `ldscope` was a dynamic executable that required
glibc to run and thus would not run on musl libc systems.

`ldscope` is built from [`src/scope_static.c`](../src/scope_static.c)
and does the following:

1. Determine `${libdirbase}`, the base directory our library directory will be under. By
   default, `/tmp` is used and this can be overriden with the `-f` comand-line
   option; i.e. `ldscope -f path ...`.
2. Create the `${libdir}/libscope-${version}/` directory if it doesn't already
   exist.
3. Extract the [Dynamic Launcher](#dynamic-launcher) into
   `${libdir}/ldscopedyn` and the shared library into `${libdir}/libscope.so`.
4. Check if we're in a musl libc environment by scanning the ELF structure of a
   stock executable (`/bin/cat`) to get the path for the dynamic linker/loader
   it's using (from the `PT_INTERP` header) and looking for `musl` in that
   path. If we find it:
    * Scan the ELF structure of the extracted `ldscopdyn` executable to get the
      dynamic linker/loader it's expecting. It'll be the glibc loader from our
      build environment; something like `/lib64/ld-linux-x86-64.so.2`. That's
      not available on the musl libc system. So...
    * Create a softlink in our library directory that points to the one we got
      from `/bin/cat`; i.e. `${libdir}/ld-linux-x86-64.so.2 -> /lib/ld-musl-x86_64.so.1`.
5. Exec `ldscopedyn` passing it the same executable arguments `ldscope` was
   passed along with `--attach` and `--library` options required.. If we found
   we are on a musl libc system in the previous step, we set
   `LD_LIBRARY_PATH=${libdir}` in the exec environment too; we're tricking the
   musl system into using it's own loader even though `ldscopedyn` wants gnu's
   loader.

## Dynamic Launcher

We refer to the `${libdir}/ldscopedyn` binary that was extracted by the [Static
Launcher](#static-launcher) as the "Dynamic Launcher". It is ultimately
responsible for ensuring the desired process is running with our library loaded
and initialized. Here's what it does:

1.  Initial setup, command-line processing, etc.
2.  [Attach](#attach)) and exit if the `--attach PID` option was given.
3.  Check if the executable to scope is static or dynamic and if it's a Go program.
4.  If it's dynamic, set `LD_PRELOAD`, exec it, wait for it to exit, and exit.
5.  The executable is static so if it's not Go, we fail politely by just
    exec'ing the executable. We're bailing out here because we currently can't
    scope a non-Go static executable.
6.  It's a static Go program so we `dlopen()` the shared library to get
    `sys_exec()` and use that to launch the executable.

## CLI

`scope` is the CLI front-end for AppScope. It conbines an "easy button" way to
use AppScope with a number of peripheral tools to access and analyze the
resulting data.

When `scope` is used to scope a new program:

1. it sets up a new _session_ directory under `~/.scope/history/` and generates
   a `scope.yml` config in there.
2. it runs `ldscope` passing through the command-line of the program to be
   scoped along with the `SCOPE_CONF_PATH` environment variable set to the full
   path of the config.

The [Static Launcher](#static-launcher) takes over from there as described
earlier. The CLI has a command-line option that add the `-f ${libdirbase}`
option.

## `LD_PRELOAD`

The simplest way to scope a program is using the `LD_PRELOAD` environment
variable but it only works for dynamic executables. Setting
`LD_PRELOAD=.../libscope.so` then running the program to be scoped tells the
dynamic loader to load our library into memory first and use the symbols it
defines first before searching other libraries. 

When our library is loaded, its constructor is called automatically. We setup
our message queues and periodic thread in that constructor and we're ready to
go.

## Attach

Attaching to an existing process instead of launching a new one is slightly
different. The [CLI](#cli)'s `scope attach PID` command sets up the session
directory the same way it does for `scope run ...` then runs the [Static
Launcher](#static-launcher) with the `--attach PID` option. Not much different
there.

The [Static Launcher](#static-launcher) simply passes the flag through to the
[Dynamic Launcher](#dynamic-launcher). Nothing different there.

The [Dynamic Launcher](#dynamic-launcher) sees the `--attach PID` option and:

1. loops through the program's shared objects looking for `libc.so` in the
   name to get its base address in memory.
2. uses `dlsym()` to find the address of `dlopen()` in the library.
3. gets the base address of the same library in the attached process using its
   `/proc/PID/maps` file.
4. uses the offset of `dlopen()` in our copy of libc to calculate the address of
   `dlopen()` in the attached process' copy of thge library.
5. uses ptrace() to do some voodoo that causes the library to be `dlload()`'ed
   in the attached process and we're done.

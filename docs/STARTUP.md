# Cribl AppScope - Startup Logic

> Warning: this document is a WIP and needs review.

On startup, AppScope needs to detect and handle a number of different
scenarios. This document is an attempt to capture this logic.

There are three ways to start a _scoped_ process:

* Using the [CLI](#cli); i.e. `scope run foo`
* Using the [Static Launcher](#static-launcher); i.e. `ldscope foo`
* Using [`LD_PRELOAD`](#ld_preload); i.e. `LD_PRELOAD=libscope.so foo`

There are two ways to _scope_ and existing process:

* Using the [CLI](#cli); i.e. `scope attach PID`
* Using the [Static Launcher](#static-launcher); i.e. `ldscope --attach PID`

## Library Directory

We need a "Library Directory" where we can put things needed for various
startup scenarios. We need a separate one for each version of AppScope too.

`${libdirbase}` is the base directory where we will create the Library
Directory.  It defaults to `/tmp` and may be overriden with command-line
options to the [CLI](#cli) or [Static Launcher](#static-launcher). Startup fails
if `${libdirbase}` doesn't exist or we can't create the Library Directory under
it.

`${libdir}` is the Library Directory; `${libdirbase}/libscope-${version}`.

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
3. Extract the [Dynamic Launcher](#dynamic-launcher) into `${libdir}/ldscopedyn`.
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
5. Exec `ldscopedyn` passing it the same arguments `ldscope` was passed. If we
   found we are on a musl libc system in the previous step, we set
   `LD_LIBRARY_PATH=${libdir}` in the exec environment too; we're tricking the
   musl system into using it's own loader even though `ldscopedyn` wants gnu's
   loader.

## Dynamic Launcher

We refer to the `${libdir}/ldscopedyn` binary that was extracted by the [Static
Launcher](#static-launcher) as the "Dynamic Launcher". It is ultimately
responsible for ensuring the desired process is running with our library loaded
and initialized.

There are a number of edge cases the logic here needs to deal with. Here's the gist:

1.  `initFn()` populates `g_fn` with pointers to all the libc functions we interpose.
2.  setenv `SCOPE_PID_ENV` to the current PID
3.  `setup_libscope()` extracts the shared library into shared memory.
4.  handle `--help` option and exit if set
5.  handle `--attach PID` option and exit if set (see [Attach](#attach))
6.  `open()` and `mmap()` the program to be run to get to the ELF headers
7.  setenv `SCOPE_APP_TYPE` to `go` for a Go program; otherwise, to `native`
8.  setenv to limit HTTP to HTTP/1.1 for a Go program
9.  if the program to run is a dynamic executable
    * setenv `LD_PRELOAD=...`
    * setenv `SCOPE_EXEC_TYPE=dynamic`
    * fork
        * in the child, exec the program and it runs using [`LD_PRELOAD`](#ld_preload)
        * in the parent, `waitpid()` the child then `release_libscope()` and exit
10. setenv `SCOPE_EXEC_TYPE=static`
11. if `LD_PRELOAD` is set, something's not working so we just unset it and
    fail-safe by exec'ing the program.
12. if program is not Go, we don't yet know how to scope non-Go statics so
    fail-safe again by exec'ing the program.
13. `dlopen()` the library we loaded into shared memory earlier and get
    `sys_exec()` with `dlsym()`
14.  exec the program using the `sys_exec()` routine which handles,
     reinitializing the library, and starting the program.

## CLI

`scope` is the CLI front-end for AppScope. It conbines an "easy button" way to
use AppScope with a number of peripheral tools to access and analyze the
resulting data.

When `scope` is used to scope a program:

1. it sets up a new _session_ folder in `~/.scope/history/` and generates a
   `scope.yml` config in that new folder.
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

1. iterate through the program's shared objects looking for `libc.so` in the
   name to get it's base address in memory.
2. use `dlsym()` to find the address of `dlopen()` in the library.
3. get the base address of the same library in the attached process using its
   `/proc/PID/maps` file.
4. use the offset of `dlopen()` in our copy of libc to calculate the address of
   `dlopen()` in the attached process' copy of thge library.
5. use ptrace() to do some voodoo that causes the library to be `dlload()`'ed
   int he attached process and we're done.


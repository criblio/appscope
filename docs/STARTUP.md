# Cribl AppScope – Startup Logic

AppScope needs to detect and handle a number of different scenarios on startup. This document attempts to capture the logic involved.

There are three ways to start a new "scoped" process:

* Using the [CLI](#cli): `scope run foo`
* Using the [Static Loader](#static-loader): `ldscope foo`
* Using [`LD_PRELOAD`](#ld_preload): `LD_PRELOAD=libscope.so foo`

There are two ways to scope an existing process:

* Using the [CLI](#cli): `scope attach PID`
* Using the [Static Loader](#static-loader): `ldscope --attach PID`

We'll cover how each approach works in the sections that follow.

## Library Directory

We need a "Library Directory" where we can put things needed for various startup scenarios. We need a separate one for each version of AppScope, too.

`${libdirbase}` is the base directory where we will create the Library Directory. It defaults to `/tmp`, but you can override this with command-line options to the [CLI](#cli) or [Static 
Loader](#static-loader). Startup fails if `${libdirbase}` doesn't exist, or if we can't create the Library Directory under it.

`${libdir}` is our Library Directory: `${libdirbase}/libscope-${version}`.

## Static Loader

We refer to `ldscope` as the "Static Loader." It's responsible ultimately for launching the [Dynamic Loader](#dynamic-loader). Since we introduced musl libc support in v.0.7, `ldscope` is a static executable and can run without libc. (Prior to v.0.7, `ldscope` was a dynamic executable that required glibc to run, thus the static loader would not run on musl libc systems.)

`ldscope` is built from [`src/scope_static.c`](../src/scope_static.c) and does the following:

1. Determine `${libdirbase}`, the base directory under which our library directory will reside. By default, `/tmp` is used. This can be overridden with the `-f` command-line option: `ldscope -f path ...`.
2. Create the `${libdir}/libscope-${version}/` directory, if it doesn't already exist.
3. Extract the [Dynamic Loader](#dynamic-loader) into `${libdir}/ldscopedyn`, and extract the shared library into `${libdir}/libscope.so`.
4. Check if we're in a musl libc environment. It does this by scanning the ELF structure of a stock executable (`/bin/cat`), to get the path for the dynamic linker/loader it's using from the `PT_INTERP` header. It then looks for `musl` in that path. If we find it:
    * Scan the ELF structure of the extracted `ldscopdyn` executable to get the dynamic linker/loader it's expecting. It'll be the glibc loader from our build environment – something like `/lib64/ld-linux-x86-64.so.2`. That's not available on the musl libc system. So...
    * Create a softlink in our library directory that points to the linker/loader we got  from `/bin/cat`; i.e. `${libdir}/ld-linux-x86-64.so.2 -> /lib/ld-musl-x86_64.so.1`.

5. Execute `ldscopedyn`, passing it the same executable arguments that we passed to `ldscope`, along with required `--attach` and `--library` options. If we found we are on a musl libc system in the previous step, we set `LD_LIBRARY_PATH=${libdir}` in the exec environment, too. We're tricking the musl system into using its own loader, even though `ldscopedyn` wants gnu's loader.
1. 
## Dynamic Loader

We refer to the `${libdir}/ldscopedyn` binary that was extracted by the [Static Loader](#static-loader) as the "Dynamic Loader." It is ultimately responsible for ensuring that the desired process is running, with our library loaded and initialized. Here's the decision tree it follows:

1.  Initial setup, command-line processing, etc.
2.  [Attach](#attach) and exit, if the `--attach PID` option was given.
3.  Check if the executable to scope is static or dynamic, and if it's a Go program.
4.  If it's dynamic, set `LD_PRELOAD`, exec it, wait for it to exit, and exit.
5.  The executable is static, so if it's not Go, we fail politely by just exec'ing the executable. We're bailing out here because we currently can't scope a non-Go static executable.
6.  It's a static Go program, so we `dlopen()` the shared library to get `sys_exec()`, and use that to launch the executable.

## CLI

`scope` is the CLI front end for AppScope. It combines an "easy button" way to
use AppScope with a number of peripheral tools to access and analyze the
resulting data.

When `scope` is used to scope a new program:

1. It sets up a new _session_ directory under `~/.scope/history/`, and generates a `scope.yml` config in there.
2. It runs `ldscope`, passing through the command-line of the program to be scoped, along with the `SCOPE_CONF_PATH` environment variable set to the full path of the config.

The [Static Loader](#static-loader) takes over from there, as described earlier. The CLI has a command-line option that add the `-f ${libdirbase}` option.

## `LD_PRELOAD`

The simplest way to scope a program is using the `LD_PRELOAD` environment variable, but this works only for dynamic executables. Setting `LD_PRELOAD=.../libscope.so`, and then running the program to be scoped, tells the dynamic loader to load our library into memory first – and use the symbols it defines first – before searching other libraries. 

When our library is loaded, its constructor is called automatically. We set up our message queues and periodic thread in that constructor, and we're ready to go.

## Attach

Attaching to an existing process instead of launching a new one is slightly
different. The [CLI](#cli)'s `scope attach PID` command sets up the session
directory the same way it does for `scope run ...` then runs the [Static
Loader](#static-loader) with the `--attach PID` option. Not much different
there.

The [Static Loader](#static-loader) simply passes the flag through to the
[Dynamic Loader](#dynamic-loader). Nothing different there.

The [Dynamic Loader](#dynamic-loader) sees the `--attach PID` option and:

1. loops through the program's shared objects looking for `libc.so` in the
   name to get its base address in memory.
2. uses `dlsym()` to find the address of `dlopen()` in the library.
3. gets the base address of the same library in the attached process using its
   `/proc/PID/maps` file.
4. uses the offset of `dlopen()` in our copy of libc to calculate the address of
   `dlopen()` in the attached process' copy of the library.
5. uses ptrace() to do some voodoo that causes the library to be `dlload()`'ed
   in the attached process and we're done.

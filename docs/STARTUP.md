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

`${libdirbase}` is the base directory where we will create the Library Directory. It defaults to `/tmp`, but you can override this with command-line options to the [CLI](#cli) or [Static Loader](#static-loader). Startup fails if `${libdirbase}` doesn't exist, or if we can't create the Library Directory under it.

`${libdir}` is our Library Directory: `${libdirbase}/libscope-${version}`.

## Static Loader

We refer to `ldscope` as the "Static Loader." It's responsible ultimately for launching the [Dynamic Loader](#dynamic-loader). Since we introduced musl libc support in v.0.7, `ldscope` is a static executable and can run without libc. (Prior to v.0.7, `ldscope` was a dynamic executable that required glibc to run, thus the loader would not run on musl libc systems.)

`ldscope` is built from [`src/scope_static.c`](../src/scope_static.c) and does the following:

1. Determine `${libdirbase}`, the base directory under which our library directory will reside. By default, `/tmp` is used. This can be overridden with the `-f` command-line option: `ldscope -f path ...`.
2. Create the `${libdir}/libscope-${version}/` directory, if it doesn't already exist.
3. Extract the [Dynamic Loader](#dynamic-loader) into `${libdir}/ldscopedyn`, and extract the shared library into `${libdir}/libscope.so`.
4. Check if we're in a musl libc environment. It does this by scanning the ELF structure of a stock executable (`/bin/cat`), to get the path for the dynamic linker/loader it's using from the `.interp` segment. It then looks for `musl` in that path. If we find it, it's using musl libc:
    * Scan the ELF structure of the extracted `ldscopdyn` executable to get the dynamic linker/loader it's expecting. It'll be the glibc loader from our build environment – something like `/lib64/ld-linux-x86-64.so.2`. That's not available on the musl libc system. So...
    * Create a softlink in our Library Directory that points to the linker/loader we got  from `/bin/cat`; i.e. `${libdir}/ld-linux-x86-64.so.2 -> /lib/ld-musl-x86_64.so.1`.
    * Modify the extracted `ldscopedyn` binary rewriting the `.interp` section to use the musl linker/loader. See `readelf -p .interp ldscopedyn` before and after to see this change.
5. Execute `ldscopedyn`, passing it the same executable arguments that we passed to `ldscope`, along with required `--attach` and `--library` options. If we found we are on a musl libc system in the previous step, we set `LD_LIBRARY_PATH=${libdir}` in the exec environment, too. We're tricking the musl system into using its own loader, even though `ldscopedyn` wants gnu's loader.

## Dynamic Loader

We refer to the `${libdir}/ldscopedyn` binary that was extracted by the [Static Loader](#static-loader) as the "Dynamic Loader." It is ultimately responsible for ensuring that the desired process is running, with our library loaded and initialized. Here's the decision tree it follows:

1. Initial setup, command-line processing, etc.
2. [Attach](#attach) and exit, if the `--attach PID` option was given.
3. Check if the executable to scope is static or dynamic, and if it's a Go program.
4. If it's dynamic, set `LD_PRELOAD`, exec it, wait for it to exit, and exit.
5. The executable is static, so if it's not Go, we fail politely by just exec'ing the executable. We're bailing out here because we currently can't scope a non-Go static executable.
6. It's a static Go program, so we `dlopen()` the shared library to get `sys_exec()`, and use that to launch the executable.

## CLI

`scope` is the CLI front end for AppScope. It combines an "easy button" way to use AppScope with a number of peripheral tools to access and analyze the resulting data.

When `scope` is used to scope a new program:

1. It sets up a new _session_ directory under `~/.scope/history/`, and generates a `scope.yml` config in there.
2. It runs `ldscope`, passing through the command-line of the program to be scoped, along with the `SCOPE_CONF_PATH` environment variable set to the full path of the config.

The [Static Loader](#static-loader) takes over from there, as described earlier. The CLI has a command-line option that add the `-f ${libdirbase}` option.

## `LD_PRELOAD`

The simplest way to scope a program is using the `LD_PRELOAD` environment variable, but this works only for dynamic executables. Setting `LD_PRELOAD=.../libscope.so`, and then running the program to be scoped, tells the dynamic loader to load our library into memory first – and use the symbols it defines first – before searching other libraries. 

When our library is loaded, its constructor is called automatically. We set up our message queues and periodic thread in that constructor, and we're ready to go.

## Attach

Attaching to an existing process instead of launching a new one is slightly different. Starting with the CLI:

1. A user runs `scope attach PID` to attach to the existing process. 
   * The CLI outputs an error and exits if the user isn't root, if PTRACE isn't enabled, or if the PID is invalid. _(TODO: Fail if process is already scoped.)_
    * The session folder the CLI creates goes under `/tmp` instead of `~root/.scope/history` where it would be with `scope run`. Permissions on that session folder are 0777 instead of 0755 too.
    * The CLI then sets `SCOPE_CONF_PATH=${session}/scope.yml` in the envronment and exec's `ldscope --attach PID [-f DIR]`. The `-f DIR` part of the command is only included if the CLI was run with `-l DIR` or `--librarypath DIR`.
2. The [Static Loader](#static-loader) gets the `--attach PID` option, the `SCOPE_CONF_PATH` environment variable, and the `-f DIR` option (optionally).
    * It extracts `ldscopedyn` and `libscope.so` to the Library Directory honoring the overridden base directory if it gets `-f DIR`. No change here.
    * Since `--attach PID` was passed in, it creates `/dev/shm/scope_${PID}.yml` and writes the contents of the file that `SCOPE_CONFIG_PATH` points to into it.
    * It continues as described in step 4 of [Static Loader](#static-loader) to detect and support a musl libc environment if detected.
    * It exec's `ldscopedyn --attach PID -l ${libdir}/scope.so`.
3. The [Dynamic Loader](#dynamic-loader) gets `--attach PID` and `-l ${libdir}/scope.so` passed in on the command line.
    * To inject the library into the target process, we need first to find the address of `dlopen()` in the libc it's using. We can safely assume the offest of the same function in the libc we're using is the same so we:
        * find the address of `libc.so` or `ld-musl` using `dl_iterate_phdr()`
        * get the address of `dlopen()` using `dlsym()`
        * find the address of `libc.so` or `ld-musl` using `/proc/${PID}/maps`
        * calculate the address of `dlopen()` in the target process
    * Next, comes the voodoo to cause the target process to load our library:
        * `ptrace(ATTACH)` to attach and temporarily stop it
        * `ptrace(GETREGS)` to backup the registers
        * Scan its `/proc/${PID}/mmap` to find a block of executable memory
        * Backup a chunk of that executable memory with ptrace(PEEKTEXT)
        * Write the string name of the library to load on the the memory block with ptrace(POKETEXT).
        * Write assembly instructions into the memory block to call a function and to break when it returns using ptrace(POKETEXT) again.
        * Setup the registers to call the `dlopen()` we found with the right arguments using ptrace(SETREGS).
        * Restart the process with ptrace(CONT) and wiat for it to get back to the breakpoint we set.
        * `ptrace(GETREGS)` again to get the return code from `dlopen()`
        * Restore the memory we munged and the registers and `ptrace(CONT)`.
        * The results of the call to `dlopen()` are in the registers we copied out at the end. If it was successful, the library is in the target process and its constructor was called.
4. The constructor in `libscope.so` has logic to find the `scope.yml` file it should use. We use `/dev/shm/scope_${PID}.yml` first followed by the normal options. See `ldescope --help config`.

If the [Static Loader](#static-loader) is run directly without using the CLI, the `SCOPE_CONF_PATH` environment variable may not be set. In that case, we check if `SCOPE_HOME` is set, we'll use `${SCOPE_HOME}/scope.yml` to populate `/dev/shm/scope_${PID}.yml`. If neither environemnt variable is set, `ldscope` doesn't create the config in shared memory at all. The library in the scoped process looks in the other normal places for configs.

> TODO: Should `ldscope` use the same logic to find a suitable `scope.yml` as `libscope.so` or is the quick-and-dirty fallback described above what we want?

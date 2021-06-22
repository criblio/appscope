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

Our Library Directory defaults to: `/tmp/libscope-${version}`. You can override this default using command-line options with the CLI or Static Loader. Below, we'll refer to your configured Library Directory path as `${libdir}`.

## Static Loader

We refer to `ldscope` as the "Static Loader." It's responsible ultimately for launching the [Dynamic Loader](#dynamic-loader). Since we introduced musl libc support in v.0.7, `ldscope` is a static executable and can run without libc. (Prior to v.0.7, `ldscope` was a dynamic executable that required glibc to run, thus the loader would not run on musl libc systems.)

`ldscope` is built from [`src/scope_static.c`](../src/scope_static.c) and does the following:

1. Determine `${libdirbase}`, the base directory under which our Library Directory will reside. By default, `/tmp` is used. This can be overridden with the `-f` command-line option: `ldscope -f DIR ...`.
2. Create the `${libdir}/libscope-${version}/` directory, if it doesn't already exist.
3. Extract the [Dynamic Loader](#dynamic-loader) into `${libdir}/ldscopedyn`, and extract the shared library into `${libdir}/libscope.so`.
4. Check if we're in a musl libc environment. It does this by scanning the ELF structure of a stock executable (`/bin/cat`), to get the path for the dynamic linker/loader it's using from the `.interp` segment. It then looks for `musl` in that path. If we find it, it's using musl libc:
    * Scan the ELF structure of the extracted `ldscopdyn` executable to get the dynamic linker/loader it's expecting. It'll be the glibc loader from our build environment – something like `/lib64/ld-linux-x86-64.so.2`. That's not available on the musl libc system. So...
    * Create a softlink in our Library Directory that points to the linker/loader we got  from `/bin/cat`; i.e. `${libdir}/ld-linux-x86-64.so.2 -> /lib/ld-musl-x86_64.so.1`.
    * Modify the extracted `ldscopedyn` binary rewriting the `.interp` section to use the musl linker/loader. Run `readelf -p .interp ldscopedyn` before and after to see this change.
5. Exec `ldscopedyn`, passing it the `--attach PID` option or executable arguments that we passed to `ldscope`. If we found we are on a musl libc system in the previous step, we set `LD_LIBRARY_PATH=${libdir}` in the exec environment, too. `ldscopedyn` has been adjusted to use musl libc's dynamic linker/loader but the soft-link and `LD_LIBRARY_PATH` are still needed for when the linker/loader goes looking for glibc symbols.

## Dynamic Loader

We refer to the `${libdir}/ldscopedyn` binary that was extracted by the [Static Loader](#static-loader) as the "Dynamic Loader." It is ultimately responsible for ensuring that the desired process is running, with our library loaded and initialized. Here's the decision tree it follows:

1. Initial setup, command-line processing, etc.
2. [Attach](#attach) and exit, if the `--attach PID` option was given.
3. Check if the executable to scope is static or dynamic, and if it's a Go program.
4. If it's dynamic:
    * set `LD_PRELOAD`
    * exec the executable with the given args
    * wait for it to exit
    * exit returning the same result as the child
4. if `LD_PRELOAD` is set:
    * unset `LD_PRELOAD`
    * re-exec `ldscopedyn` with the same arguments
> TODO: What scenario is this handling?

5. If it's not a Go program:
    * we fail politely by just exec'ing the executable.
    * We're bailing out here because we currently can't scope a non-Go static executable. This may change in the future.
6. Finally, it's a static Go program, so:
    * we `dlopen()` our library and get `sys_exec()` which is a version of the standard `exec()` function except that it doens't nuke the current process before loading the new one. We use this to keep our library loaded.
    * use `sys_exec()` to exec the executable 

## CLI

`scope` is the CLI front end for AppScope. It combines an "easy button" way to use AppScope with a number of peripheral tools to access and analyze the resulting data.

When `scope` is used to scope a new program:

1. It sets up a new _session_ directory under `~/.scope/history/`, and generates a `scope.yml` config in there.
2. It runs `ldscope`, passing through the command-line of the program to be scoped, along with the `SCOPE_CONF_PATH` environment variable set to the full path of the config.

The [Static Loader](#static-loader) takes over from there, as described earlier. The CLI has a command-line option that adds the `-f ${libdirbase}` option.

## `LD_PRELOAD`

The simplest way to scope a program is using the `LD_PRELOAD` environment variable, but this works only for dynamic executables. Setting `LD_PRELOAD=.../libscope.so`, and then running the program to be scoped, tells the dynamic loader to load our library into memory first – and use the symbols it defines first – before searching other libraries. 

When our library is loaded, its constructor is called automatically. We set up our interpositions, message queues, and periodic thread in that constructor, and we're ready to go.

## Attach

Attaching to an existing process instead of launching a new one is slightly different. Starting with the CLI:

1. A user runs `scope attach PID` to attach to the existing process. 
    * The CLI outputs an error and exits if the user isn't root, if PTRACE isn't enabled, if the PID is invalid, or if process is already "scoped".
    * The session folder is created under `/tmp` instead of `~root/.scope/history` where it would be with `scope run`. Permissions on session directories and files are 0777 and 0666 instead of the normal 0755 and 0644. This could be a security concern in some scenarios so we need to make sure we note this somewhere like an "Are you sure?" prompt or a blurb in the release notes.
    * The CLI then sets `SCOPE_CONF_PATH=${session}/scope.yml` in the envronment and exec's `ldscope --attach PID [-f DIR]`. The `-f DIR` part of the command is only included if the CLI was run with `-l DIR` or `--librarypath DIR`.

2. The [Static Loader](#static-loader) gets the `--attach PID` option, the `SCOPE_CONF_PATH` environment variable, and the `-f DIR` option (optionally).
    * It extracts `ldscopedyn` and `libscope.so` to the Library Directory honoring the overridden base directory if it gets `-f DIR`. No change here.
    * Since `--attach PID` was passed in
        * Ensure we are running as root and the PID exists
        * Create `/dev/shm/scope_attach_${PID}.env` containing environment variables that should be set by the library once attached; `SCOPE_CONFIG_PATH`, `SCOPE_HOME`, SCOPE_EXEC_PATH`, and `SCOPE_LIB_PATH`.
    * It continues as described in step 4 of [Static Loader](#static-loader) to detect and support a musl libc environment if detected.
    * It exec's `${libdir}/ldscopedyn --attach PID` with `SCOPE_LIB_PATH=${libdir}/libscope.so` added to its environment variables.

3. The [Dynamic Loader](#dynamic-loader) gets `--attach PID` on the command line and `SCOPE_LIB_PATH` in the environment.
    * To inject the library into the target process, we need first to find the address of `dlopen()` in the libc it's using. We can safely assume the offest of that function in the libc we're using is the same in the target process so:
        * find the `libc.so` or `ld-musl` library in the current process using `dl_iterate_phdr()`
        * get the address of `dlopen()` in it using `dlsym()`
        * find the address of the same library in the target procerss using `/proc/${PID}/maps`
        * calculate the address of `dlopen()` in the target process
    * Next, comes the magic to cause the target process to load our library:
        * `ptrace(ATTACH)` to attach to the target process and temporarily stop all threads in it
        * `ptrace(GETREGS)` to backup its registers
        * Scan its `/proc/${PID}/mmap` to find a block of executable memory
        * Backup a chunk of that executable memory with ptrace(PEEKTEXT)
        * Write the full path to our library into the the memory block with ptrace(POKETEXT).
        * Write assembly instructions into the memory block to call a function and [break](https://en.wikipedia.org/wiki/INT_(x86_instruction)) when it returns using ptrace(POKETEXT) again.
        * Setup the registers for that function call to call the `dlopen()` we found for our library using ptrace(SETREGS).
        * Restart the process with ptrace(CONT) and wait for it to get back to the breakpoint we set.
          * If the return code indicated it succeeded, our library is now loaded and intialized in the target process.
        * `ptrace(GETREGS)` again to get the return code from `dlopen()`
        * Restore the memory we used and the registers then `ptrace(CONT)` to let the program continue where it was.

4. In the attached process, the library is loaded and the constructor is run
    * The `/dev/shm/scope_attach_${pid}.yml` exists so we load configs from that and remove the file.
    * The presence if the config in shared memory indicates to the library that it was loaded via "attach" so it needs to use GOT hooking for the interposed functions.

5. There a couple of relevant behaviours in the library:
    * The constructor's looks for `/dev/shm/scope_attach_${PID}.env` and applies the environment variables there.
    * The presence of that file indicates we've been injected and need to handle the hooking differently.

> Note: You will see term "inject" used in the code. It's synonomous with "attach".

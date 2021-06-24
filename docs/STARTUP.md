# AppScope – Startup Logic

> Note: This document has a narrow focus on the *interaction between* AppScope components. To learn *what makes AppScope work*, such as how AppScope interposes functions and interacts with applications and operating systems, see the [AppScope docs](https://appscope.dev/docs/) on the website.

AppScope need a place to unpack components it uses at runtime. We create a directory named `libscope-${version}`, the _Library Directory_, for each version of AppScope that gets run. It's created in `/tmp` by default but this base directory can be overridden with the `-l DIR` command-line option. This doc refers to the base directory (`/tmp`) as `${libdirbase}` and the Library Directory (`${libdirbase}/libscope-0.7.0`) as `${libdir}`.

## Startup Scenarios

Upon startup, AppScope detects and handles a number of different scenarios according to the logic outlined here.

Each scenario is a variation on one of two themes:

- Attaching AppScope to a new process.
- Attaching AppScope to a running process.

And, every scenario is an interaction between some combination of AppScope components:

- `libscope`, the AppScope library — the core component of AppScope.
- `ldscopedyn`, the Dynamic Loader, which ensures that the desired process is running with the AppScope library loaded and initialized. Some, but not all, Dynamic Loader scenarios entail setting the `LD_PRELOAD` environment variable.
- `ldscope`, the Static Loader, which launches the Dynamic Loader in some, but not all, scenarios.
- `scope`, the CLI, the AppScope front end.

Each scenario description in this section provides a basic command pattern. Subsequent sections detail the logic of the components involved.

To start and scope a new process (`foo` in the examples below), you can use:

- The [CLI](#cli): `scope run foo`
- The [Static Loader](#static-loader), for static or dynamic executables: `ldscope foo`
- The [`LD_PRELOAD`](#ld_preload) environment variable, for dynamic executables only: `LD_PRELOAD=libscope.so foo`

To scope a running process (whose process ID is `PID` in the examples below), you can use:

- The [CLI](#cli): `scope attach PID`
- The [Static Loader](#static-loader), for static or dynamic executables: `ldscope --attach PID`

## CLI

`scope` is the CLI front end for AppScope. Besides being the "easy button" way to use AppScope, the CLI provides tools for accessing and analyzing the resulting data.

When you scope a new program, `scope`:

1. Sets up a new _session_ directory under `~/.scope/history/`.
1. In the _session_ directory, generates a `scope.yml` config file.
1. Sets the `SCOPE_CONF_PATH` environment variable to the full path of the `scope.yml` config file.  
1. Runs `ldscope`, passing through the command-line of the program to scope, along with `SCOPE_CONF_PATH`.
   The [Static Loader](#static-loader) takes over from there, as described in the next section.

If you need to tell `scope` where `${libdirbase}` is, use the `-l ${libdirbase}` option.

## Static Loader

`ldscope` is built from [`src/scope_static.c`](../src/scope_static.c). It is responsible for launching the [Dynamic Loader](#dynamic-loader). It does the following:

1. Determine `${libdirbase}` using either the default `/tmp` or the value provided in the `-f DIR` command-line option.
1. Create the `${libdir}` Library Directory, `${libdirbase}/libscope-${version}/`, if it doesn't already exist.
1. Extract the [Dynamic Loader](#dynamic-loader) into `${libdir}/ldscopedyn`, and extract the shared library into `${libdir}/libscope.so`.
1. Determine whether we are in a musl libc environment. To do this, `ldscope` scans the ELF structure of a stock executable (`/bin/cat`), and gets the path for the dynamic linker/loader it's using, from the `.interp` segment. If it's `ld-musl`, we're in a musl libc environment. This requires some extra steps to ensure that AppScope uses the correct linker/loader:
    - Scan the ELF structure of the extracted `ldscopdyn` executable to get the dynamic linker/loader it's expecting. This will be the glibc loader from the build environment – something like `/lib64/ld-linux-x86-64.so.2`. That's not available on the musl libc system. So...
    - Create a softlink in the Library Directory that points to the linker/loader we got from `/bin/cat`; e.g., `${libdir}/ld-linux-x86-64.so.2 -> /lib/ld-musl-x86_64.so.1`.
    - Modify the extracted `ldscopedyn` binary, rewriting the `.interp` section to use the musl linker/loader. Run `readelf -p .interp ldscopedyn` before and after to see this change.
1. Finally, exec `ldscopedyn` with the arguments we were started with (minus `-f DIR`) and `LD_LIBRARY_PATH=${libdir}` added to the environment if on musl libc.

## Dynamic Loader

The `${libdir}/ldscopedyn` binary extracted by the [Static Loader](#static-loader) is called the Dynamic Loader. It is ultimately responsible for ensuring that the desired process is running, with our library loaded and initialized. 

The Dynamic Loader's decision tree looks like this:

1. Initial setup, command-line processing, etc.
1. [Attach](#attach) and exit, if the `--attach PID` option was given.
1. Determine whether the executable to scope is static or dynamic, and whether it's a Go program.
1. If it's dynamic:
    - Set `LD_PRELOAD`.
    - Exec the executable with the given args.
    - Wait for it to exit.
    - Exit, returning the same result as the child.
1. If `LD_PRELOAD` is set:
    - Unset `LD_PRELOAD`.
    - Re-exec `ldscopedyn` with the same arguments.
1. If it's not a Go program:
    - Fail politely: Just exec the executable.
    - AppScope can't scope a non-Go static executable, so it must bail out here. This may change in the future.
1. If we reach this point, it's a static Go program, so:
    - `dlopen()` our library and get `sys_exec()`.
    - Use `sys_exec()` to exec the executable without nuking the current process.

## `LD_PRELOAD`

Setting `LD_PRELOAD=.../libscope.so`, then running the program to be scoped, tells the kernel's linker/loader to load our library into memory first – and use the symbols it defines first – before searching other libraries.

When our library is loaded, its constructor is called automatically. AppScope sets up its interpositions, message queues, and periodic thread in that constructor, and we're ready to go.

## Attach to a Running Process

Attaching to a running process differs from launching a new one. This scenario begins with the CLI, then involves the Static Loader and the Dynamic Loader in turn. Note: In the code, the term *inject* is synonomous with *attach*.

1. A user runs `scope attach PID` to attach to a running process.
1. The CLI does the following:
    - Exit with an error if any of the following are true: The user isn't root; PTRACE isn't enabled; the PID is invalid; or, the process is already "scoped".
    - Create the session folder is under `/tmp` (instead of `~root/.scope/history` where it would be with `scope run`). Permissions on session directories and files are 0777 and 0666 instead of the normal 0755 and 0644.
    - Set `SCOPE_CONF_PATH=${session}/scope.yml` in the envronment.
    - Exec `ldscope --attach PID [-f DIR]`. The `-f DIR` part of the command is only included if the CLI was run with `-l DIR` or `--librarypath DIR`.
1. The [Static Loader](#static-loader) gets the `--attach PID` option, the `SCOPE_CONF_PATH` environment variable, and the `-f DIR` option (optionally), and then does the following:
    - Extract `ldscopedyn` and `libscope.so` to the Library Directory honoring the overridden base directory if it gets `-f DIR`. No change here.
    - Ensure we are running as root and the PID exists
    - Create `/dev/shm/scope_attach_${PID}.env` containing environment variables that should be set by the library once attached: `SCOPE_CONFIG_PATH`, `SCOPE_HOME`, `SCOPE_EXEC_PATH`, and `SCOPE_LIB_PATH`.
    - Detect whether we are in a musl libc environment, and if so, support that environment as described in the [Static Loader](#static-loader) section above.
    - Exec `${libdir}/ldscopedyn --attach PID` with `SCOPE_LIB_PATH=${libdir}/libscope.so` added to its environment variables.
1. The [Dynamic Loader](#dynamic-loader) gets `--attach PID` on the command line and `SCOPE_LIB_PATH` in the environment.
    - To inject the library into the target process, the Dynamic Loader needs to make it `dlopen()` our library so:
        - Find the `libc.so` or `ld-musl` library in the current process, using `dl_iterate_phdr()`.
        - Get the address of `dlopen()` in that library, using `dlsym()`.
        - Find the address of the same library in the target process, using `/proc/${PID}/maps`.
        - Use the offset of `dlopen()` in our process' libc to calcualte the address of `dlopen()` in the target process.
    - Next, to cause the target process to load our library, Dynamic Loader does the following:
        - Attach to the target process and temporarily stop all its threads, with `ptrace(ATTACH)`.
        - Back up the process's registers, with `ptrace(GETREGS)`.
        - Scan its `/proc/${PID}/mmap` to find a block of executable memory.
        - Back up a chunk of that executable memory, with `ptrace(PEEKTEXT)`.
        - Write our library's full path into the the memory block, with `ptrace(POKETEXT)`.
        - Write assembly instructions into the memory block to call a function and [break](https://en.wikipedia.org/wiki/INT_(x86_instruction)), using `ptrace(POKETEXT)` again.
        - Set up the registers for that function call to be the `dlopen()` we found earlier, using `ptrace(SETREGS)`.
        - Restart the process with `ptrace(CONT)`, and wait for it to get back to the breakpoint we set.
        - Get the return code from `dlopen()`, with `ptrace(GETREGS)`.
        - Restore the memory we used and the registers, then `ptrace(CONT)` to let the program continue where it was.
        - If the return code indicated success, our library was loaded and initialized in the target process.
    - Last, remove the `/dev/shm/scope_attach_${pid}.env` file.
1. In the attached process, the library was loaded and the constructor was run.
    - The `/dev/shm/scope_attach_${pid}.env` was loaded and corresponding environment variables were set.
    - The presence of the .env file caused the library to setup the interpositions using GOT hooking.

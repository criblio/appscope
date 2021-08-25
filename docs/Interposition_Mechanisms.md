# Interposition Mechanisms

In a previous blog, we outlined how AppScope uses function interpositioning as a means to extract information from applications, in user mode, at run time. You can check it out [here](https://docs.google.com/document/d/1FrN_POjjLzmqZZI2tYOwKv9MYFxW_wIZd75f5Pf2hbQ/edit?usp=sharing). In this blog, we want to provide an overview of a few (among many available) interposition mechanisms that we&#39;ve found valuable in building AppScope. This blog delves into application development details, and will be of particular interest to developers who love to maximize their apps&#39; performance.

## Library Preload

Library preloading is a feature of a modern dynamic linker/loader (`ld.so`). A dynamic linking and loading capability is available on most Unix-based and Windows systems. The linker/loader&#39;s preload feature allows a user-specified shared library to be loaded before all other shared libraries required by an executable.

The dynamic linker resolves external addresses for an executable, using the symbol namespace of libraries as they are loaded. It loads the symbol namespace in library load order. Therefore, if a library includes a function named `fwrite`, the symbol table will include an entry for `fwrite`. The address for `fwrite` is determined when the library is loaded, and the linker uses that address to resolve references to `fwrite`.

Now assume that an application uses `fwrite`. It has a dependency on the `libc.so` library because that is where the function `fwrite` is resolved when the application is built. The dynamic loader will resolve the address to `fwrite` when `libc.so` is loaded. Now, if a library is preloaded before `libc.so`, and the preloaded library defines a function `fwrite`, the dynamic linker will resolve `fwrite`&#39;s address to the preloaded library instead of using the address to `fwrite` in `libc.so`. The function `fwrite` has just been _interposed_. It is now up to the interposed `fwrite` function to locate the address of `fwrite` in `libc.so`, so that it can (in turn) call `libc.so:fwrite`.

Library preloading has been around for a long time, with multiple uses. Library preload is most commonly used to replace the memory allocation subsystem. For example, Valgrind uses this mechanism to track memory leaks.

There are several alternatives for memory allocators supporting the definitions of `malloc`, `alloc`, and `free`. Many of these are in common use. The `ptmalloc2` subsystem is the default memory allocator used by `glibc`. Chromium replaces `ptmalloc2` with `tcmalloc`, a Google-defined memory allocator subsystem. The `jemalloc` subsystem has been used with FreeBSD, and has found its way into numerous applications that rely on its predictable behavior.

You too could implement your own `malloc(3)` and `free(3)` functions, with which you could perform leak checking or memory access control. In this case, the library to be preloaded would implement the functions you want to interpose.

Note that only functions of dynamically loaded libraries can be interposed by means of library preload. Library preload would not be used to interpose functions that an application itself provides, nor would it be used to interpose any functions in a statically linked executable.

A few references:

- [https://google.github.io/tcmalloc/overview.html](https://google.github.io/tcmalloc/overview.html)
- [https://github.com/emeryberger/Malloc-Implementations/tree/master/allocators/ptmalloc/ptmalloc2](https://github.com/emeryberger/Malloc-Implementations/tree/master/allocators/ptmalloc/ptmalloc2)
- [http://jemalloc.net/#:~:text=jemalloc%20is%20a%20general%20purpose,rely%20on%20its%20predictable%20behavior](http://jemalloc.net/#:~:text=jemalloc%20is%20a%20general%20purpose,rely%20on%20its%20predictable%20behavior).
- [https://man7.org/linux/man-pages/man8/ld.so.8.html](https://man7.org/linux/man-pages/man8/ld.so.8.html)

## GOT Hooking 

GOT stands for Global Offsets Table, and this topic gets very low-level and detailed very quickly. We&#39;ll try to keep it simple, to describe in what conditions this is needed for interposing functions.

PLT stands for Procedure Linkage Table. The dynamic linker uses the PLT to enable calling of external functions. The complete address resolution is accomplished by the dynamic linker when an executable is loaded. The Global Offsets Table is used to resolve addresses in conjunction with the PLT. At the risk of oversimplification, PLT is the code that gets executed when an external function is called, while the GOT is data that defines the actual function address.

The dynamic loader uses what&#39;s known as lazy binding. By convention, when the dynamic linker loads a library, it will put an identifier and a resolution function into known places in the GOT. Then, the first call to an external function uses a call to a default stub in the PLT. The PLT loads the identifier and calls into the dynamic linker. The linker resolves the address of the function being called, and the associated GOT entry is updated. The next time the PLT entry is called, it will load the actual address of the function from the GOT, rather than the dynamic loader lookup. Very cool!

In order to interpose a function, the entry in the GOT is saved and replaced with the address of a function which will perform the interposition.

Why and where is this needed? Given the use of library preloading, GOT hooking is used sparingly and in specific scenarios. For the most part, again at the risk of oversimplification, GOT hooking is used when an application loads libraries itself, after init and independent of the dynamic loader. The Python interpreter loads libraries in support of import statements. Apache loads libraries as a part of module management, as defined in configuration files. GOT hooking can be used to interpose functions deployed in any of these application-loaded libraries.

A few references:

- [https://www.technovelty.org/linux/plt-and-got-the-key-to-code-sharing-and-dynamic-libraries.html](https://www.technovelty.org/linux/plt-and-got-the-key-to-code-sharing-and-dynamic-libraries.html)
- [http://index-of.es/Varios-2/Learning%20Linux%20Binary%20Analysis.pdf](http://index-of.es/Varios-2/Learning%20Linux%20Binary%20Analysis.pdf)
- [https://atakua.org/old-wp/wp-content/uploads/2015/03/libelf-by-example-20100112.pdf](https://atakua.org/old-wp/wp-content/uploads/2015/03/libelf-by-example-20100112.pdf)
- [https://systemoverlord.com/2017/03/19/got-and-plt-for-pwning.html](https://systemoverlord.com/2017/03/19/got-and-plt-for-pwning.html)
- [http://lj.rossia.org/users/herm1t/78510.html](http://lj.rossia.org/users/herm1t/78510.html)

## Function Hooking

We noted that library preload is not useful for interposing functions in the application itself, nor in statically linked executables. The same applies to GOT hooking. The vast majority of executables are dynamic, where the above techniques work fine.

The most common examples of static executables (although certainly not the only examples) are Go applications. These are the places where function hooking comes into play.

Function hooking is accomplished by modifying _code that implements_ the function to be interposed. Again, this approach is not new. And again at the risk of oversimplification, function hooking works by placing a `jmp` instruction in a function preamble. The destination of the `jmp` causes the interposed function to be called before the original function is called.It involves a lot of assembly language. It depends on the hardware architecture and instruction definition. When done right, it works rather well.

In order to update code, function hooking must write to memory in executable space. This, of course, is not normally enabled. Therefore, a system call (such as `mprotect` in Unix/Linux systems) is required.

Changing permissions in memory is accomplished on page boundaries. The page associated with the function to be interposed is given write permissions; the code is modified with the `jmp` instruction; and write permissions are removed. As long as functionality associated with `mprotect` is possible, function hooking can work well.

Loading a static executable and ensuring that the library code we use is also loaded, sets up a topic for a different blog. There are cases where it is desirable to interpose internal application functions. These are relatively rare, but useful when needed.

A few references:

- [https://refspecs.linuxfoundation.org/elf/elf.pdf](https://refspecs.linuxfoundation.org/elf/elf.pdf)
- [https://github.com/kubo/funchook](https://github.com/kubo/funchook)
- [https://software.intel.com/content/www/us/en/develop/articles/introduction-to-x64-assembly.html](https://software.intel.com/content/www/us/en/develop/articles/introduction-to-x64-assembly.html)
- [http://jbremer.org/x86-api-hooking-demystified/](http://jbremer.org/x86-api-hooking-demystified/)


// The macro interpose takes 3 params
// the name of the global variable as defined above ex: go_hook_write
// the name of the Go function that will be called
// the name of the C function interposed handler
.macro interpose a_func, c_func
.global \a_func
.hidden \c_func
\a_func:
    // This creates a stack frame of 0x20 bytes
    // The add grows the stacks by 0x18
    // The push decrements the stack pointer, $rsp, by 8
    // then stores the register, the stack frame then is
    // 0x18 + 0x08 = 0x20
    push %rax
    sub $0x18, %rsp

    // Call the 'C' handler with the current stack pointer
    // as argument, e.g.:
    // go_write(stackptr)
    mov %rsp, %r11
    add $0x20, %r11
    mov %r11, 0x08(%rsp)
    lea \c_func@GOTPCREL(%rip), %r11
    mov (%r11), %r11
    mov %r11, (%rsp)

    // We call a 'C' function by passing the function
    // address and a param to the Go function cgocall.
    // Function: syscall.asmcgocall(fnc, arg)
    // cgocall (or asmcgocall) switches from the Go stack
    // to what the Go runtime calls a system stack
    // then calls fnc and returns a value. We are
    // using the fact that the return value remains in %rax.
    // The return value is also on the stack. We may need to
    // extract it from the stack if Go syscall code changes.
    lea go_runtime_cgocall@GOTPCREL(%rip), %r11
    mov (%r11), %r11
    mov (%r11), %r11
    call *%r11
    mov %rax, %r11

    # housekeeping
    add $0x18, %rsp
    pop %rax

    // The value returned from our 'C' handler function is always
    // the address of the trampoline that gets us back to the
    // original Go function. The correct trampoline address is
    // created by funchook_prepapre().
    jmp *%r11
.endm

.text
    interpose go_hook_write, go_write
    interpose go_hook_open, go_open
    interpose go_hook_unlinkat, go_unlinkat
    interpose go_hook_getdents, go_getdents
    interpose go_hook_socket, go_socket
    interpose go_hook_accept4, go_accept4
    interpose go_hook_read, go_read
    interpose go_hook_close, go_close
    interpose go_hook_tls_read, go_tls_read
    interpose go_hook_tls_write, go_tls_write
    //interpose go_hook_readResponse, go_readResponse
    //interpose go_hook_pc_write, go_pc_write
    interpose go_hook_exit, go_exit
    interpose go_hook_die, go_die

/*
    Debug hint:
    When you need to change the assy code, 
    it's a pain to step through the code
    in macro form. What has worked much better
    is to copy/paste the code from the macro
    and execute it as one of the interposition
    handlers. The code below is a replica of
    what's in the macro with comments removed
    and labels changed (if any).

    In order to use this as go_hook_write, for
    example,
    1) uncomment the code below
    2) change go_hook_write_1 to go_hook_write
    3) comment the macro instance above
       #interpose go_hook_write, go_write
    4) build
*/

/*
.global go_hook_pc_write
go_hook_pc_write:
    push %rax
    sub $0x18, %rsp

    //mov %rsp, %r11
    //add $0x20, %r11
    //mov %r11, 0x08(%rsp)
    lea go_pc_write@GOTPCREL(%rip), %r11
    mov (%r11), %r11
    //mov %r11, (%rsp)

    //lea go_runtime_cgocall@GOTPCREL(%rip), %r11
    //mov (%r11), %r11
    //mov (%r11), %r11
    call *%r11
    mov %rax, %r11

    add $0x18, %rsp
    pop %rax
    jmp *%r11
*/

.global go_hook_pc_write
go_hook_pc_write:
    mov %rsp, %r11
    push %rax
    push %rbx

    mov %r11, %rbx
    lea go_pc_write@GOTPCREL(%rip), %r11
    mov (%r11), %r11
    mov %r11, %rax

    lea go_runtime_cgocall_regs@GOTPCREL(%rip), %r11
    mov (%r11), %r11
    mov (%r11), %r11
    call *%r11
    mov %rax, %r11

    pop %rbx
    pop %rax
    jmp *%r11


.global go_hook_readResponse
go_hook_readResponse:
    mov %rsp, %r11
    push %rax
    push %rbx

    mov %r11, %rbx
    lea go_readResponse@GOTPCREL(%rip), %r11
    mov (%r11), %r11
    mov %r11, %rax

    lea go_runtime_cgocall_regs@GOTPCREL(%rip), %r11
    mov (%r11), %r11
    mov (%r11), %r11
    call *%r11
    mov %rax, %r11

    pop %rbx
    pop %rax
    jmp *%r11

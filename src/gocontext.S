#include <sys/syscall.h>
#include <asm/prctl.h>

#include "gocontext.h"

.global	go_hook_write, go_hook_open, go_hook_socket, go_hook_accept4
.global go_hook_read, go_hook_close, go_hook_tls_read, go_hook_tls_write

// The macro interpose takes 3 params
// the name of the global variable as defined above ex: go_hook_write
// the name of the Go function that will be called
// the name of the C function interposed handler
.macro interpose a_func, c_func
\a_func:
    // This creates a stack frame of 0x20 bytes
    // The add grows the stacks by 0x18
    // The push decrements the stack pointer, $rsp, by 8
    // then stores the register, the stack frame then is
    // 0x18 + 0x08 = 0x20
    sub $0x18, %rsp
    push %rax

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
    pop %rax
    add $0x18, %rsp

    // The value returned from our 'C' handler function is always
    // the address of the trampoline that gets us back to the
    // original Go function. The correct trampoline address is
    // created by funchook_prepapre().
    jmp *%r11
.endm

.text
    interpose go_hook_write, go_write
    interpose go_hook_open, go_open
    interpose go_hook_socket, go_socket
    interpose go_hook_accept4, go_accept4
    interpose go_hook_read, go_read
    interpose go_hook_close, go_close
    interpose go_hook_tls_read, go_tls_read
    interpose go_hook_tls_write, go_tls_write
/*
go_hook_write_1:
    sub $0x18, %rsp
    push %rax

    mov %rsp, %r11
    add $0x20, %r11
    mov %r11, 0x08(%rsp)
    lea go_write@GOTPCREL(%rip), %r11
    mov (%r11), %r11
    mov %r11, (%rsp)

    lea go_runtime_cgocall@GOTPCREL(%rip), %r11
    mov (%r11), %r11
    mov (%r11), %r11
    call *%r11
    mov %rax, %r11

    pop %rax
    add $0x18, %rsp

    jmp *%r11
*/

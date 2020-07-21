#include <sys/syscall.h>
#include <asm/prctl.h>

#include "gocontext.h"

.global	go_hook_write, go_hook_open, go_hook_socket, go_hook_accept4
.global go_hook_read, go_hook_close, do_go

// The macro interpose takes 3 params
// the name of the global variable as defined above ex: go_hook_write
// the name of the Go function that will be called
// the name of the C function interposed handler
.macro interpose a_func, go_func, c_func
\a_func:
    # put args on the stack for the cgo call
    # fnc is the C handler go_new_store()
    # arg is a pointer to the stack
    sub $0x20,%rsp
    mov %fs:0xfffffffffffffff8, %r11
    push %r11
    lea go_new_store@GOTPCREL(%rip), %r11
    mov (%r11), %r11
    push %r11

    # call cgo
    # runtime.asmcgocall(go_new_store, stackpointer)
    lea go_runtime_cgocall@GOTPCREL(%rip), %r11
    mov (%r11), %r11
    mov (%r11), %r11
    call *%r11

    # check the return value
    test %rax, %rax
    jle out_\a_func

    # housekeeping
    pop %r10
    pop %r10
    add $0x20,%rsp

    # save the return address
    # we have an entry to save the return addr given to us from go_new_store
    mov (%rsp), %r11
    mov %r11, 0x08(%rax)

    # change the return addr on the Go stack to come back here
    movabs $toc_\a_func, %r11
    mov %r11, (%rsp)

    # do the Go function
    lea \go_func@GOTPCREL(%rip), %r11
    mov (%r11), %r11
    mov (%r11), %r11
    jmp *%r11

toc_\a_func:
    # restore the original return address
    sub $0x20,%rsp
    mov %fs:0xfffffffffffffff8, %r11
    push %r11
    lea go_reset_store@GOTPCREL(%rip), %r11
    mov (%r11), %r11
    push %r11

    # call cgo to execute go_reset_store(stackpointer)
    lea go_runtime_cgocall@GOTPCREL(%rip), %r11
    mov (%r11), %r11
    mov (%r11), %r11
    call *%r11

    # check the return value
    test %rax, %rax
    jle out_\a_func

    # get the return address
    mov 0x08(%rax), %r11

    # housekeeping
    pop %r10
    pop %r10
    add $0x20,%rsp

    # restore the original return address
    # must be a push in order to keep the stack right
    push %r11
    mov %rsp, %r11

    # put args on the stack for the cgo call
    sub $0x20,%rsp
    push %r11
    lea \c_func@GOTPCREL(%rip), %r11
    mov (%r11), %r11
    push %r11

    # call cgo c_func(stackpointer)
    lea go_runtime_cgocall@GOTPCREL(%rip), %r11
    mov (%r11), %r11
    mov (%r11), %r11
    call *%r11

out_\a_func:
    pop %r11
    pop %r11
    add $0x20,%rsp
    ret

.endm

.text
    interpose go_hook_write, go_syscall_write, go_write
    interpose go_hook_open, go_syscall_open, go_open
    interpose go_hook_socket, go_syscall_socket, go_socket
    interpose go_hook_accept4, go_syscall_accept4, go_accept4
    interpose go_hook_read, go_syscall_read, go_read
    interpose go_hook_close, go_syscall_close, go_close

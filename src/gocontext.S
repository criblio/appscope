#include <sys/syscall.h>
#include <asm/prctl.h>

#include "gocontext.h"

.global	go_hook_write, go_hook_open, go_hook_socket, go_hook_read, do_go
.global go_hook_entersyscall, go_hook_exitsyscall, go_hook_morestack_noctxt
    
.text
go_hook_entersyscall:
    retq

go_hook_exitsyscall:
    retq

go_hook_morestack_noctxt:
    retq

go_hook_write:
    # put args on the stack for the cgo call
    sub $0x20,%rsp
    mov %fs:0xfffffffffffffff8, %r11
    push %r11
    lea go_new_store@GOTPCREL(%rip), %r11
    mov (%r11), %r11
    push %r11

    # call cgo
    lea go_runtime_cgocall@GOTPCREL(%rip), %r11
    mov (%r11), %r11
    mov (%r11), %r11
    call *%r11

    test %rax, %rax
    jle write_out
    # housekeeping
    pop %r10
    pop %r10
    add $0x20,%rsp

    # save the return address
    mov (%rsp), %r11
    mov %r11, 0x08(%rax)

    # change the return addr on the Go stack to come back here
    movabs $write_toc, %r11
    mov %r11, (%rsp)

    # do the Go function
    lea go_syscall_write@GOTPCREL(%rip), %r11
    mov (%r11), %r11
    mov (%r11), %r11
    jmp *%r11

write_toc:
    # restore the original return address
    sub $0x20,%rsp
    mov %fs:0xfffffffffffffff8, %r11
    push %r11
    lea go_reset_store@GOTPCREL(%rip), %r11
    mov (%r11), %r11
    push %r11

    # call cgo
    lea go_runtime_cgocall@GOTPCREL(%rip), %r11
    mov (%r11), %r11
    mov (%r11), %r11
    call *%r11

    test %rax, %rax
    jle write_out

    # get the return address
    mov 0x08(%rax), %r11

    # housekeeping
    pop %r10
    pop %r10
    add $0x20,%rsp

    # restore the original return address
    push %r11
    mov %rsp, %r11

    # put args on the stack for the cgo call
    sub $0x20,%rsp
    push %r11
    lea go_write@GOTPCREL(%rip), %r11
    mov (%r11), %r11
    push %r11

    # call cgo
    lea go_runtime_cgocall@GOTPCREL(%rip), %r11
    mov (%r11), %r11
    mov (%r11), %r11
    call *%r11

write_out:
    pop %r11
    pop %r11
    add $0x20,%rsp
    ret

    mov %rsp, %r10
    mov (%rsp), %r11
    push %rdi
    push %rsi
    push %rdx
    push %rax
    mov %r11, %rdi
    mov %r10, %rsi
    mov %fs:0xfffffffffffffff8,%rdx

    lea go_write@GOTPCREL(%rip), %r11
    mov (%r11), %r11
    call *%r11

    pop %rax
    pop %rdx
    pop %rsi
    pop %rdi

    # save the return address
    lea go_stack_saves@GOTPCREL(%rip), %r11
    mov (%r11), %r11
    mov (%rsp), %r10
    mov %r10, (%r11)

    # change return addr on the Go stack to come back here
    movabs $write_ret, %r11
    mov %r11, (%rsp)

    # do the Go function
    lea go_syscall_write@GOTPCREL(%rip), %r11
    mov (%r11), %r11
    mov (%r11), %r11
    jmp *%r11

write_ret:
    # restore the original return address
    lea go_stack_saves@GOTPCREL(%rip), %r11
    mov (%r11), %r11
    mov (%r11), %r11
    push %r11

    # tell the lib code where we came from
    lea go_what@GOTPCREL(%rip), %r11
    mov (%r11), %r11
    movl $SYSCALL_WRITE, (%r11)
    jmp do_go

go_hook_open:
    # put args on the stack for the cgo call
    sub $0x20,%rsp
    mov %fs:0xfffffffffffffff8, %r11
    push %r11
    lea go_new_store@GOTPCREL(%rip), %r11
    mov (%r11), %r11
    push %r11

    # call cgo
    lea go_runtime_cgocall@GOTPCREL(%rip), %r11
    mov (%r11), %r11
    mov (%r11), %r11
    call *%r11

    test %rax, %rax
    jle open_out
    # housekeeping
    pop %r10
    pop %r10
    add $0x20,%rsp

    # save the return address
    mov (%rsp), %r11
    mov %r11, 0x08(%rax)

    # change the return addr on the Go stack to come back here
    movabs $open_toc, %r11
    mov %r11, (%rsp)

    # do the Go function
    lea go_syscall_open@GOTPCREL(%rip), %r11
    mov (%r11), %r11
    mov (%r11), %r11
    jmp *%r11

open_toc:
    # restore the original return address
    sub $0x20,%rsp
    mov %fs:0xfffffffffffffff8, %r11
    push %r11
    lea go_reset_store@GOTPCREL(%rip), %r11
    mov (%r11), %r11
    push %r11

    # call cgo
    lea go_runtime_cgocall@GOTPCREL(%rip), %r11
    mov (%r11), %r11
    mov (%r11), %r11
    call *%r11

    test %rax, %rax
    jle write_out

    # get the return address
    mov 0x08(%rax), %r11

    # housekeeping
    pop %r10
    pop %r10
    add $0x20,%rsp

    # restore the original return address
    push %r11
    mov %rsp, %r11

    # put args on the stack for the cgo call
    sub $0x20,%rsp
    push %r11
    lea go_open@GOTPCREL(%rip), %r11
    mov (%r11), %r11
    push %r11

    # call cgo
    lea go_runtime_cgocall@GOTPCREL(%rip), %r11
    mov (%r11), %r11
    mov (%r11), %r11
    call *%r11

open_out:
    pop %r11
    pop %r11
    add $0x20,%rsp
    ret

# Manual stack switch
    # save the return address
    lea go_stack_saves@GOTPCREL(%rip), %r11
    mov (%r11), %r11
    mov (%rsp), %r10
    mov %r10, (%r11)

    # change return addr on the Go stack to come back here
    movabs $open_ret, %r11
    mov %r11, (%rsp)

    # do the Go function
    lea go_syscall_open@GOTPCREL(%rip), %r11
    mov (%r11), %r11
    mov (%r11), %r11
    jmp *%r11

open_ret:
    # restore the original return address
    lea go_stack_saves@GOTPCREL(%rip), %r11
    mov (%r11), %r11
    mov (%r11), %r11
    push %r11

    # get the result from the open call
    lea go_results@GOTPCREL(%rip), %r11
    mov (%r11), %r11
    #mov %rax, (%r11)
    mov 0x30(%rsp), %r10
    mov %r10, (%r11)

    # tell the lib code where we came from
    lea go_what@GOTPCREL(%rip), %r11
    mov (%r11), %r11
    movl $SYSCALL_OPEN, (%r11)
    jmp do_go

go_hook_socket:
    # put args on the stack for the cgo call
    sub $0x20,%rsp
    mov %fs:0xfffffffffffffff8, %r11
    push %r11
    lea go_new_store@GOTPCREL(%rip), %r11
    mov (%r11), %r11
    push %r11

    # call cgo
    lea go_runtime_cgocall@GOTPCREL(%rip), %r11
    mov (%r11), %r11
    mov (%r11), %r11
    call *%r11

    test %rax, %rax
    jle socket_out
    # housekeeping
    pop %r10
    pop %r10
    add $0x20,%rsp

    # save the return address
    mov (%rsp), %r11
    mov %r11, 0x08(%rax)

    # change the return addr on the Go stack to come back here
    movabs $socket_toc, %r11
    mov %r11, (%rsp)

    # do the Go function
    lea go_syscall_socket@GOTPCREL(%rip), %r11
    mov (%r11), %r11
    mov (%r11), %r11
    jmp *%r11

socket_toc:
    # restore the original return address
    sub $0x20,%rsp
    mov %fs:0xfffffffffffffff8, %r11
    push %r11
    lea go_reset_store@GOTPCREL(%rip), %r11
    mov (%r11), %r11
    push %r11

    # call cgo
    lea go_runtime_cgocall@GOTPCREL(%rip), %r11
    mov (%r11), %r11
    mov (%r11), %r11
    call *%r11

    test %rax, %rax
    jle socket_out

    # get the return address
    mov 0x08(%rax), %r11

    # housekeeping
    pop %r10
    pop %r10
    add $0x20,%rsp

    # restore the original return address
    push %r11
    mov %rsp, %r11

    # put args on the stack for the cgo call
    sub $0x20,%rsp
    push %r11
    lea go_socket@GOTPCREL(%rip), %r11
    mov (%r11), %r11
    push %r11

    # call cgo
    lea go_runtime_cgocall@GOTPCREL(%rip), %r11
    mov (%r11), %r11
    mov (%r11), %r11
    call *%r11

socket_out:
    pop %r11
    pop %r11
    add $0x20,%rsp
    ret


go_hook_read:
    # save the return address
    lea go_stack_saves@GOTPCREL(%rip), %r11
    mov (%r11), %r11
    mov (%rsp), %r10
    mov %r10, (%r11)

    # change return addr on the Go stack to come back here
    movabs $read_ret, %r11
    mov %r11, (%rsp)

    # do the Go function
    lea go_syscall_read@GOTPCREL(%rip), %r11
    mov (%r11), %r11
    mov (%r11), %r11
    jmp *%r11

read_ret:
    # restore the original return address
    lea go_stack_saves@GOTPCREL(%rip), %r11
    mov (%r11), %r11
    mov (%r11), %r11
    push %r11

    # tell the lib code where we came from
    lea go_what@GOTPCREL(%rip), %r11
    mov (%r11), %r11
    movl $SYSCALL_READ, (%r11)
    jmp do_go

do_go:
    # save the Go stack
    lea go_stack@GOTPCREL(%rip), %r11
    mov (%r11), %r11
    mov %rsp, (%r11)

    # Get the Go params
    lea go_params@GOTPCREL(%rip), %r11
    mov (%r11), %r11
    mov 8(%rsp), %r10
    mov %r10, 0(%r11)
    mov 16(%rsp), %r10
    mov %r10, 8(%r11)
    mov 24(%rsp), %r10
    mov %r10, 16(%r11)

    # Switch to the Scope stack
    lea scope_stack@GOTPCREL(%rip), %r11
    mov (%r11), %r11
    mov (%r11), %rsp

    # Save regs
    push %rdi
    push %rsi
    push %rdx
    push %rcx
    push %r8
    push %r9
    
    # save the Go TLS addr
    lea go_fs@GOTPCREL(%rip), %r11
    mov (%r11), %rsi
    mov $ARCH_GET_FS, %edi
    mov $SYS_arch_prctl, %eax
    syscall
	test %rax, %rax
    jz got_arch
    hlt             # change this!

got_arch:
    # Call a Scope function with void param
    lea go_hook_test@GOTPCREL(%rip), %r11
    mov (%r11), %r11
    call *%r11

    # Restore args
    pop %r9
    pop %r8
    pop %rcx
    pop %rdx
    pop %rsi
    pop %rdi

    # Switch back to the Go stack
    lea go_stack@GOTPCREL(%rip), %r11
    mov (%r11), %r11
    mov (%r11), %rsp
    ret


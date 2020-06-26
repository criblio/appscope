#include <sys/syscall.h>
#include <asm/prctl.h>
    
.global	go_hook_write
.global	go_hook_open
    
.text
go_hook_write:
    lea go_what@GOTPCREL(%rip), %r11
    mov (%r11), %r11
    movl $1, (%r11)
    jmp do_go

go_hook_open:
    lea go_what@GOTPCREL(%rip), %r11
    mov (%r11), %r11
    movl $2, (%r11)
    jmp do_go

do_go:
    # save the Go stack
    lea go_stack@GOTPCREL(%rip), %r11
    mov (%r11), %r11
    mov %rsp, (%r11)

    # save the Go TLS addr
    lea go_fs@GOTPCREL(%rip), %r11
    mov (%r11), %rsi
    mov $ARCH_GET_FS, %edi
    mov $SYS_arch_prctl, %eax
    syscall
	test	%rax, %rax
    jz got_arch
    hlt
    
    # Get the Go params
got_arch:
    lea go_params@GOTPCREL(%rip), %r11
    mov (%r11), %r11
    mov 8(%rsp), %r10
    mov %r10, 0(%r11)
    mov 16(%rsp), %r10
    mov %r10, 8(%r11)
    mov 24(%rsp), %r10
    mov %r10, 16(%r11)

    # lock
    lea runtime_lock@GOTPCREL(%rip), %r11
    mov (%r11), %r11
    mov (%r11), %r11
#    call *%r11

    # Switch to the Scope stack
    lea scope_stack@GOTPCREL(%rip), %r11
    mov (%r11), %r11
    mov (%r11), %rsp
#    push %rbp
#    mov %rsp,%rbp
#    sub $0x50,%rsp

    # Save regs
    push %rdi
    push %rsi
    push %rdx
    push %rcx
    push %r8
    push %r9
    
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
#    pop %rbp
    
    # Switch back to the Go stack
    lea go_stack@GOTPCREL(%rip), %r11
    mov (%r11), %r11
    mov (%r11), %rsp

    # Do the Go function
    lea go_what@GOTPCREL(%rip), %r11
    mov (%r11), %r11
    mov (%r11), %eax
    cmp $0x1,%eax
    je  do_write
    cmp $0x2,%eax
    je  do_open
    jmp do_error

do_write:
    lea go_syscall_write@GOTPCREL(%rip), %r11
    mov (%r11), %r11
    mov (%r11), %r11
    jmp *%r11

do_open:
    lea go_syscall_open@GOTPCREL(%rip), %r11
    mov (%r11), %r11
    mov (%r11), %r11
    jmp *%r11

do_error:
    hlt



#    lea go_syscall_write@GOTPCREL(%rip), %rax
#    mov (%rax), %rax
#    mov (%rax), %rax
#    jmp *%rax

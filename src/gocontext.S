#define g_m 0x30
#define m_g0 0x0
#define m_gsignal 0x50
#define g_sched_sp 0x38 //0x40 in 8
#define g_sched_pc 0x40
#define g_sched_ret 0x58
#define g_sched_bp 0x68
#define m_g0_sched_sp 0x38 //0x40 in 8

/*
 * Go 17+ will pass input params and return values in registers.
 * This macro will move us onto the system stack (like asmcgocall) and push registers onto it for our usage,
 * and so that we can restore them after calling our c_func.
 * We must not push anything onto the go stack, else the stack checker will create a panic.

 * From: https://go.googlesource.com/go/+/refs/heads/dev.regabi/src/cmd/compile/internal-abi.md
 * The following registers are used for integer arguments and results:
 * RAX, RBX, RCX, RDI, RSI, R8, R9, R10, R11
 * Note that X15 is defined to be always a fixed value of zero
 */
.macro interpose_system_stack a_func, c_func, s_func
.global \a_func
.hidden \c_func
\a_func:
    endbr64

    // Is g valid?
    mov  %fs:0xfffffffffffffff8, %r12   // g
    cmp  $0x0, %r12
    je   1f

    // Is this g m->gsignal? If so, already on the system stack
    mov  g_m(%r12), %r13                // g->m
    mov  m_gsignal(%r13), %r13          // m->gsignal
    cmp  %r12, %r13
    je   1f

    // Is this g g->m->g0 (new thread)? If so, already on the system stack
    mov  g_m(%r12), %r13                // g->m
    mov  (%r13), %r13                   // m->g0
    cmp  %r12, %r13
    je   1f

    // Else switch to system stack
    // Set g state to use system stack
    lea  go_systemstack_switch@GOTPCREL(%rip), %r13
    mov  %r13, g_sched_pc(%r12)         // Store addr of system stack switch in g->sched.pc
    lea  0x8(%rsp), %r13
    mov  %r13, g_sched_sp(%r12)         // Store g stack in g->sched.sp
    movq $0x0, g_sched_ret(%r12)        // Store NULL in g->sched.ret
    mov  %rbp, g_sched_bp(%r12)         // Store g frame pointer in g->sched.bp

    mov  g_m(%r12), %r13                // g->m
    mov  (%r13), %r13                   // m->g0

    // The system stack switch
    mov  %fs:0xfffffffffffffff8, %r14   // Ensure r14 points to g in all versions
    mov  %rsp, %r12                     // r12 is g_stack (note: if making callbacks, use offset instead)
    mov  %r13, %fs:0xfffffffffffffff8   // Make current g m->g0 (The context switch)
    mov  m_g0_sched_sp(%r13), %rsp      // m->g0->sched.sp (The stack switch)

    lea  \c_func@GOTPCREL(%rip), %r13 
    mov  (%r13), %r13
    call \s_func                        // Safe to call a C func now that we are on the system stack
                                        // Registers are saved & restored in c_handler
	mov %r14, %fs:0xfffffffffffffff8    // Restore the g
	mov %r12, %rsp                      // Restore the g_stack

    jmp *%r13                           // Return through the trampoline

1:
    // The current g is using the system stack, don't switch stacks
    mov  %rsp, %r12                     // save g stack for use in the C func
    lea \c_func@GOTPCREL(%rip), %r13
    mov  (%r13), %r13
    call \s_func
    jmp  *%r13
.endm

.text
c_handler:
    // Inputs:
    // r12 is the caller stack
    // r13 is c_func address
    // r14 is the caller g
    // Note: r14 is not set in the case where a stack switch isn't needed
    //       It's ok here becasue we don't use it to restore the g
    //       However, don't rely on r14 in C code
    // Returns:
    // r13 is the trampoline address

    // Save regs on the stack
    push %r14                            // g
    push %r12                            // gstack
    push %r11
    push %r10
    push %r9     
    push %r8    
    push %rsi  
    push %rdi 
    push %rdx
    push %rcx
    push %rbx
    push %rax 

    mov  %rsp, %rdi                      // C func param
    call *%r13

    // Handle return state and restore stack
    // Save the return through trampoline addr from the C func
    mov %rax, %r13

    // Restore saved registers
    pop  %rax
	pop  %rbx
	pop  %rcx
	pop  %rdx
    pop  %rdi
    pop  %rsi
    pop  %r8
    pop  %r9
    pop  %r10
    pop  %r11
    pop  %r12                            // g stack
    pop  %r14                            // g

    ret
 
c_syscall_handler:
    // Inputs:
    // r12 is the caller stack
    // r13 is c_func address
    // r14 is the caller g
    // Returns:
    // r13 is the trampoline address

    // Save regs on the stack
    push %rax                            // linux syscall num
    push %r14                            // g
    push %r12                            // g stack
    push %r11
    push %r10
    push %r9                             // linux 6th param
    push %r8                             // linux 5th param
    push %rsi                            // linux 2nd param
    push %rdi                            // linux 1st param
    push %rdx                            // linux 3rd param
    push %rcx                            // linux 4th param
    push %rbx

    syscall
    push %rax                            // rc

    mov  %rsp, %rdi                      // C func param
    call *%r13

    // Handle return state and restore stack
    // Save the return through trampoline addr from the C func
    mov %rax, %r13

    // Restore saved registers
    pop  %rax
	pop  %rbx
	pop  %rcx
	pop  %rdx
    pop  %rdi
    pop  %rsi
    pop  %r8
    pop  %r9
    pop  %r10
    pop  %r11
    pop  %r12                            // g stack
    pop  %r14                            // g
    add $0x8, %rsp                       // Adjust the stack for pushing rax twice

    add $0x2, %r13                       // Skip the syscall instruction
    ret

    interpose_system_stack go_hook_reg_syscall               go_syscall              c_syscall_handler
    interpose_system_stack go_hook_reg_rawsyscall            go_rawsyscall           c_syscall_handler
    interpose_system_stack go_hook_reg_syscall6              go_syscall6             c_syscall_handler
    interpose_system_stack go_hook_reg_tls_server_read       go_tls_server_read      c_handler
    interpose_system_stack go_hook_reg_tls_server_write      go_tls_server_write     c_handler
    interpose_system_stack go_hook_reg_tls_client_read       go_tls_client_read      c_handler
    interpose_system_stack go_hook_reg_tls_client_write      go_tls_client_write     c_handler
    interpose_system_stack go_hook_reg_http2_server_read     go_http2_server_read    c_handler
    interpose_system_stack go_hook_reg_http2_server_write    go_http2_server_write   c_handler
    interpose_system_stack go_hook_reg_http2_server_preface  go_http2_server_preface c_handler
    interpose_system_stack go_hook_reg_http2_client_read     go_http2_client_read    c_handler
    interpose_system_stack go_hook_reg_http2_client_write    go_http2_client_write   c_handler
    interpose_system_stack go_hook_exit                      go_exit                 c_handler
    interpose_system_stack go_hook_die                       go_die                  c_handler

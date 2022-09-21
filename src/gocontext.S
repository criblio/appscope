#define g_m 0x30
#define m_g0 0x0
#define m_gsignal 0x50
#define g_sched_sp 0x38
#define g_sched_pc 0x40
#define g_sched_ret 0x58
#define g_sched_bp 0x68
#define m_g0_sched_sp 0x38

// The macro interpose takes 3 params
// the name of the global variable as defined above ex: go_hook_write
// the name of the Go function that will be called
// the name of the C function interposed handler
.macro interpose a_func, c_func
.global \a_func
.hidden \c_func
\a_func:
    // grab the caller stack address before changing anything
    mov %rsp, %r11

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

.macro interpose_regs a_func, c_func
.global \a_func
.hidden \c_func
\a_func:
    endbr64
    mov %rsp, %r12
    push %rax
    push %rbx
    push %rcx
    push %rdi
    push %rsi
    push %r8
    push %r9
    push %r10
    push %r11
    sub $0x20, %rsp

    mov %r12, 0x08(%rsp)
    lea \c_func@GOTPCREL(%rip), %r12
    mov (%r12), %r12
    mov %r12, (%rsp)

    lea go_runtime_cgocall@GOTPCREL(%rip), %r12
    mov (%r12), %r12
    mov (%r12), %r12
    call *%r12
    xorps %xmm15,%xmm15
    mov %fs:0xfffffffffffffff8,%r14

    mov %rax, %r12

    add $0x20, %rsp
    pop %r11
    pop %r10
    pop %r9
    pop %r8
    pop %rsi
    pop %rdi
    pop %rcx
    pop %rbx
    pop %rax

    jmp *%r12
.endm

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
.macro interpose_system_stack a_func, c_func, idx
.global \a_func
.hidden \c_func
\a_func:
    endbr64

//TODO if syscall call syscall_check

    mov  %fs:0xfffffffffffffff8, %r12   // g

    //  g == 0; a thread is starting or exiting; It's using the system stack
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
    mov  %r13, g_sched_pc(%r12)         // store addr of system stack switch in g->sched.pc
    lea  0x8(%rsp), %r13
    mov  %r13, g_sched_sp(%r12)         // store g stack in g->sched.sp
    movq $0x0, g_sched_ret(%r12)        // store NULL in g->sched.ret
    mov  %rbp, g_sched_bp(%r12)         // store g frame pointer in g->sched.bp

    mov  g_m(%r12), %r13                // g->m
    mov  (%r13), %r13                   // m->g0

    // The system stack switch
    mov  %fs:0xfffffffffffffff8, %r14   // Ensure r14 points to g in all versions
    mov  %rsp, %r12                     // r12 is g_stack (note: if making callbacks, use offset instead)
    mov  %r13, %fs:0xfffffffffffffff8   // Make current g m->g0 (The context switch)
    mov  m_g0_sched_sp(%r13), %rsp      // m->g0->sched.sp (The stack switch)

    lea  \c_func@GOTPCREL(%rip), %r13 
    mov  (%r13), %r13
    call c_handler                      // Safe to call a C func now that we are on the system stack
                                        // Registers are saved & restored in c_handler

	mov %r14, %fs:0xfffffffffffffff8    // Restore the g
	mov %r12, %rsp                      // Restore the g_stack

    jmp *%r13                           // Return through the trampoline

1:
    // The current g is using the system stack, don't switch stacks
    mov  %rsp, %r12
    lea \c_func@GOTPCREL(%rip), %r13
    mov  (%r13), %r13
    call c_handler
    jmp  *%r13
.endm

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
    push %r14    // caller g
    push %r12    // caller stack
    push %r11
    push %r10
    push %r9     // linux 6th param
    push %r8     // linux 5th param
    push %rsi    // linux 2nd param
    push %rdi    // linux 1st param
    push %rdx    // linux 3rd param
    push %rcx    // linux 4th param
    push %rbx
    push %rax    // linux syscall num

    mov  %rsp, %rdi                      // C func param
    call *%r13

    // Handle return state and restore stack
    // Save the return through trampoline addr from the C func
    mov %rax, %r13
    //xorps %xmm15,%xmm15
    //mov %fs:0xfffffffffffffff8, %r14

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
    pop  %r12                            // Caller stack
    pop  %r14                            // Caller g

    ret
 
/*
c_syscall_handler:
    // Inputs:
    // r12 is the caller stack
    // r13 is c_func address
    // r14 is the caller g
    // Returns:
    // r13 is the trampoline address

    // Save regs on the stack
    push %r14    // caller g
    push %r12    // caller stack
    push %r11
    push %r10
    push %r9     // linux 6th param
    push %r8     // linux 5th param
    push %rsi    // linux 2nd param
    push %rdi    // linux 1st param
    push %rdx    // linux 3rd param
    push %rcx    // linux 4th param
    push %rbx
    push %rax    // linux syscall num

  syscall
  push %rax

    mov  %rsp, %rdi                      // C func param
    call *%r13

    // Handle return state and restore stack
    // Save the return through trampoline addr from the C func
    mov %rax, %r13
    //xorps %xmm15,%xmm15
    //mov %fs:0xfffffffffffffff8, %r14

    // Restore saved registers
    pop  %rax
  add $0x8, %rsp
	pop  %rbx
	pop  %rcx
	pop  %rdx
    pop  %rdi
    pop  %rsi
    pop  %r8
    pop  %r9
    pop  %r10
    pop  %r11
    pop  %r12                            // Caller stack
    pop  %r14                            // Caller g

  add $0x2, %r13                       // Skip the syscall instruction
    ret
 
syscall_check:
    // Inputs:
    // ? is the index

    // Is this a syscall we support?
    cmp  $0x1, %rax
    je 1f
    cmp $0x101, %rax
	je 1f
	cmp $0x107, %rax
	je 1f
	cmp $0xd9, %rax
	je 1f
	cmp $0x29, %rax
	je 1f
	cmp $0x120, %rax
	je 1f
	cmp $0x0, %rax
	je 1f
	cmp $0x3, %rax
	je 1f

    // If not a supported syscall, get the return addr and done
    push %rdi
    push %rax
    mov $\idx, %rdi
    lea return_addr_idx@GOTPCREL(%rip), %r12 // not safe?
    mov (%r12), %r12
    call *%r12
    mov %rax, %r12
    pop %rax
    pop %rdi
    jmp *%r12




1:
	ret


return_to_trampoline:
    // If not a supported syscall, get the return addr and done
    mov $\idx, %r13
    lea return_addr_idx_asm@GOTPCREL(%rip), %r12 // not safe?
    mov (%r12), %r12
    call *%r12
    jmp *%r13
*/


.text
    interpose              go_hook_tls_server_read           go_tls_server_read
    interpose              go_hook_tls_server_write          go_tls_server_write
    interpose              go_hook_tls_client_read           go_tls_client_read
    interpose              go_hook_tls_client_write          go_tls_client_write
    interpose              go_hook_http2_server_read         go_http2_server_read
    interpose              go_hook_http2_server_write        go_http2_server_write
    interpose              go_hook_http2_server_preface      go_http2_server_preface
    interpose              go_hook_http2_client_read         go_http2_client_read
    interpose              go_hook_http2_client_write        go_http2_client_write
    interpose              go_hook_exit                      go_exit
    interpose              go_hook_die                       go_die

    interpose_system_stack go_hook_reg_tls_server_read       go_tls_server_read
    interpose_system_stack go_hook_reg_tls_server_write      go_tls_server_write
    interpose_system_stack go_hook_reg_tls_client_read       go_tls_client_read
    interpose_system_stack go_hook_reg_tls_client_write      go_tls_client_write
    interpose_system_stack go_hook_reg_http2_server_read     go_http2_server_read
    interpose_system_stack go_hook_reg_http2_server_write    go_http2_server_write
    interpose_system_stack go_hook_reg_http2_server_preface  go_http2_server_preface
    interpose_system_stack go_hook_reg_http2_client_read     go_http2_client_read
    interpose_system_stack go_hook_reg_http2_client_write    go_http2_client_write

    interpose_system_stack go_hook_set_inputs_g_syscall      go_set_inputs_g_syscall    0x0
    interpose_system_stack go_hook_set_inputs_g_rawsyscall   go_set_inputs_g_rawsyscall 0x1
    interpose_system_stack go_hook_set_inputs_g_syscall6     go_set_inputs_g_syscall6   0x2
    interpose_system_stack go_hook_reg_syscall               go_syscall                 0x3
    interpose_system_stack go_hook_reg_rawsyscall            go_rawsyscall              0x4
    interpose_system_stack go_hook_reg_syscall6              go_syscall6                0x5



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

/*
 * We created a second macro named with the _regs suffix that is used in
 * Go 1.17 and 1.18 interpositions. The differences from the previous
 * asmcgocall macro are slight. The first is that before we do any
 * interposition, we save the values from the registers %rax and %rbx and
 * restore them before returning to regular execution. This is because in
 * Go 1.17 and 1.18, the ABI has fundamentally changed, in that results
 * are returned in registers. The second difference is that we restore
 * the value of g into the %r14 register. In Go 1.17 and 1.18, the
 * function preamble has changed in that it always expects %r14 to
 * contain the g before entering a function (for stack checking).
 * If we don't save and restore values in these registers, they will
 * often become corrupted and result in a serious failure.
 *
 * From: https://go.googlesource.com/go/+/refs/heads/dev.regabi/src/cmd/compile/internal-abi.md
 * The following registers are used for integer arguments and results:
 * RAX, RBX, RCX, RDI, RSI, R8, R9, R10, R11
 *
 * Note that X15 is defined to be always a fixed value of zero
 */
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

.macro interpose_stack_regs a_func, c_func
.global \a_func
.hidden \c_func
\a_func:
    endbr64
    push %rax
    push %rbx
    push %rcx
    push %rdi
    push %rsi
    push %r8
    push %r9
    push %r10
    push %r11
    mov %rsp, %r12
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
  * Stub for use with a low level syscall function
  * Create a stack with input params
  * Make the syscall
  * Add to the stack with return params
  * Call asmcgocall with params for c_syscall and the created stack
  * Be sure to jump to the trampoline after the syscall which is
  * plus 2 bytes from the return value from funchook_prepare; the trampoline.
  */
.macro interpose_syscall_regs a_func, c_func
.global \a_func
.hidden \c_func
\a_func:
    endbr64

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

 // syscall

//  push %rax

    mov %rsp, %r12
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

    pop %rax
//  add $0x8, %rsp
	pop %rbx
	pop %rcx
	pop %rdx
    pop %rdi
    pop %rsi
    pop %r8
    pop %r9
    pop %r10
    pop %r11
	
//    add $0x2, %r12
    jmp *%r12
.endm

.macro interpose_syscall_lim a_func, c_func, idx
.global \a_func
.hidden \c_func
\a_func:
    endbr64
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

    // if not a supported syscall, get the return addr and done
    push %rdi
    push %rax
    mov $\idx, %rdi
    lea return_addr_idx@GOTPCREL(%rip), %r12
    mov (%r12), %r12
    call *%r12
    mov %rax, %r12
    pop %rax
    pop %rdi
    jmp *%r12

1:
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


    mov %rsp, %r12
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

    pop %rax
	pop %rbx
	pop %rcx
	pop %rdx
    pop %rdi
    pop %rsi
    pop %r8
    pop %r9
    pop %r10
    pop %r11

    jmp *%r12
.endm

.macro interpose_syscall_nostack a_func, c_func
.global \a_func
.hidden \c_func
\a_func:
    endbr64
    // get thge g
    mov  %fs:0xfffffffffffffff8, %r12

    // is g valid?
    cmp  $0x0, %r12
    je   1f

    // is this g m->gsignal? if so, already on the system stack
    mov  0x30(%r12), %r13         // g->m
    mov  0x50(%r13), %r13         // m->gsignal
    cmp  %r12, %r13
    je   1f

    // is this g g->m->g0 (new thread), if so, already on the system stack
    mov  0x30(%r12), %r13         // g->m
    mov  (%r13), %r13             // m->g0
    cmp  %r12, %r13
    je   1f

    // else switch to system stack
    // set g state to use of system stack
    lea go_systemstack_switch@GOTPCREL(%rip), %r13
    mov (%r13), %r13
    mov (%r13), %r13
    call *%r13

    mov  0x30(%r12), %r13         // g->m
    mov  (%r13), %r13             // m->g0

    // the system stack switch
    mov  %rsp, %r12               // r12 is caller stack, r14 is caller g
    mov  %r13, %fs:0xfffffffffffffff8
    mov  0x38(%r13), %rsp         // m->g0->sched.sp

    // create stack frame on the system stack
    //sub  $0x10, %rsp
    //and  $0xfffffffffffffff0, %rsp
    lea  \c_func@GOTPCREL(%rip), %r13
    mov  (%r13), %r13
    call c_handler

    mov  %r14, %fs:0xfffffffffffffff8    // restore the caller g
    mov  %r12, %rsp                      // restore the caller stack

    jmp *%r13                           // return through the trampoline

    // The curreent g is using the system stack, don't switch stacks
1:
    mov  %rsp, %r12
    lea \c_func@GOTPCREL(%rip), %r13
    mov  (%r13), %r13
    call c_handler
    jmp  *%r13
.endm

.text
    interpose go_hook_tls_server_read, go_tls_server_read
    interpose go_hook_tls_server_write, go_tls_server_write
    interpose go_hook_tls_client_read, go_tls_client_read
    interpose go_hook_tls_client_write, go_tls_client_write
    interpose go_hook_http2_server_read, go_http2_server_read
    interpose go_hook_http2_server_write, go_http2_server_write
    interpose go_hook_http2_server_preface, go_http2_server_preface
    interpose go_hook_http2_client_read, go_http2_client_read
    interpose go_hook_http2_client_write, go_http2_client_write
    interpose go_hook_exit, go_exit
    interpose go_hook_die, go_die

    interpose_regs go_hook_reg_tls_server_read, go_tls_server_read
    interpose_regs go_hook_reg_tls_server_write, go_tls_server_write
    interpose_regs go_hook_reg_tls_client_read, go_tls_client_read
    interpose_regs go_hook_reg_tls_client_write, go_tls_client_write
    interpose_regs go_hook_reg_http2_server_read, go_http2_server_read
    interpose_regs go_hook_reg_http2_server_write, go_http2_server_write
    interpose_regs go_hook_reg_http2_server_preface, go_http2_server_preface
    interpose_regs go_hook_reg_http2_client_read, go_http2_client_read
    interpose_regs go_hook_reg_http2_client_write, go_http2_client_write

    //nterpose_syscall_regs go_hook_reg_syscall go_syscall
    //interpose_syscall_regs go_hook_reg_rawsyscall go_rawsyscall
    //interpose_syscall_regs go_hook_reg_syscall6 go_syscall6

    //interpose_syscall_regs go_hook_set_inputs_g_syscall go_set_inputs_g_syscall
    //interpose_syscall_regs go_hook_set_inputs_g_rawsyscall go_set_inputs_g_rawsyscall
    //interpose_syscall_regs go_hook_set_inputs_g_syscall6 go_set_inputs_g_syscall6


    interpose_syscall_nostack go_hook_reg_syscall go_syscall
    interpose_syscall_nostack go_hook_reg_rawsyscall go_rawsyscall
    interpose_syscall_nostack go_hook_reg_syscall6 go_syscall6

    interpose_syscall_nostack go_hook_set_inputs_g_syscall go_set_inputs_g_syscall
    interpose_syscall_nostack go_hook_set_inputs_g_rawsyscall go_set_inputs_g_rawsyscall
    interpose_syscall_nostack go_hook_set_inputs_g_syscall6 go_set_inputs_g_syscall6

c_handler:
    // inputs:
    // r12 is the caller stack
    // r13 is c_func address
    // r14 is the caller g
    // returns:
    // r13 is the trampoline address

    // save regs on the stack
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
    //lea  \c_func@GOTPCREL(%rip), %r12
    call *%r13

    // handle return state and restore stack
    // save the return through trampoline addr from the C func
    mov %rax, %r13
    //xorps %xmm15,%xmm15
    //mov %fs:0xfffffffffffffff8, %r14

    // restore saved registers
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
    pop  %r12                            // caller stack
    pop  %r14                            // caller g
    // add $0x10, %rsp
    ret

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
.global go_hook_write
go_hook_write:
    push %rax
    sub $0x18, %rsp

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

    add $0x18, %rsp
    pop %rax
    jmp *%r11
*/

/*
.global go_hook_reg_tls_client_write
go_hook_reg_tls_client_write:   
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
    lea go_tls_client_write@GOTPCREL(%rip), %r12
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
*/

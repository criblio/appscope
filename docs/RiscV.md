# RISC-V

This document describe the support of RISC-V architecture in AppScope

# Emulation

https://wiki.ubuntu.com/RISC-V - see Booting with QEMU part
Improtant:
Install go via `snap`
```
sudo snap install go --classic
```
I would recommend give more memory and cpu resources

# TODO:
- [ ] Add offset support in wrap_go.c
- [ ] Add support looks_like_first_inst_of_go_func in wrap_go.c
- [ ] Add asm support in wrap_go.c
- [ ] Add asm support in gocontext
- [ ] Add asm support in com.c (pcre2_match_wrapper, regexec_wrapper)
- [ ] Add support for RISC-V in funchook library (trampoline)
- [ ] CI support ?
- [ ] Ubuntu 20.04 is the earliest release which supports RISC-V

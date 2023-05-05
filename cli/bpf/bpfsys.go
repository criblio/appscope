package bpf

import (
	"fmt"
	"runtime"
	"unsafe"

	"golang.org/x/sys/unix"
)

// Pointer wraps an unsafe.Pointer to be 64bit to
// conform to the syscall specification.
type Pointer struct {
	ptr unsafe.Pointer
}

// NewPointer creates a 64-bit pointer from an unsafe Pointer.
func NewPointer(ptr unsafe.Pointer) Pointer {
	return Pointer{ptr: ptr}
}

// bpfCmd BPF syscall commands
type bpfCmd int

// Open a file descriptor for the eBPF program corresponding to specified program id
const BPF_PROG_GET_FD_BY_ID bpfCmd = 13

// Obtain information about the eBPF object corresponding to
const BPF_OBJ_GET_INFO_BY_FD bpfCmd = 15

// unixBbpf is wrapper to BPF syscall
func unixBbpf(cmd bpfCmd, attr unsafe.Pointer, size uintptr) (uintptr, error) {
	r1, _, errno := unix.Syscall(unix.SYS_BPF, uintptr(cmd), uintptr(attr), size)
	runtime.KeepAlive(attr)
	if errno != 0 {
		return r1, fmt.Errorf("unixBbpf failed: %w", errno)
	}

	return r1, nil
}

// objGetInfoByFdAttr is anonymous structure used by BPF_OBJ_GET_INFO_BY_FD
type objGetInfoByFdAttr struct {
	bpfFd   uint32
	infoLen uint32
	info    Pointer
}

// objGetInfoByFdAttr get information about the eBPF object
func objGetInfobyFd(attr *objGetInfoByFdAttr) error {
	_, err := unixBbpf(BPF_OBJ_GET_INFO_BY_FD, unsafe.Pointer(attr), unsafe.Sizeof(*attr))
	return err
}

// progGetFdByIdAttr is structure used by BPF_PROG_GET_FD_BY_ID
type progGetFdByIdAttr struct {
	progId uint32
}

// objGetProgFdbyId a file descriptor for the eBPF proram from a program id
func objGetProgFdbyId(attr *progGetFdByIdAttr) (int, error) {
	fd, err := unixBbpf(BPF_PROG_GET_FD_BY_ID, unsafe.Pointer(attr), unsafe.Sizeof(*attr))
	return int(fd), err
}

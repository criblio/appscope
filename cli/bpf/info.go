package bpf

import (
	"runtime"
	"unsafe"
)

type Info interface {
	info() (unsafe.Pointer, uint32)
}

// objInfo retrieve information about specific eBPF object
func objInfo(fd int, info Info) error {
	ptr, len := info.info()
	err := objGetInfobyFd(&objGetInfoByFdAttr{
		bpfFd:   uint32(fd),
		infoLen: len,
		info:    NewPointer(ptr),
	})
	runtime.KeepAlive(fd)
	return err
}

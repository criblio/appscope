package bpf

// ProgFdFromId obtain the eBPF program file descriptor from eBPF program id
func ProgFdFromId(progId int) (int, error) {
	return objGetProgFdbyId(&progGetFdByIdAttr{progId: uint32(progId)})
}

package bpf

import (
	"unsafe"
)

type linkType uint32

const (
	BPF_LINK_TYPE_UNSPEC         linkType = 0
	BPF_LINK_TYPE_RAW_TRACEPOINT linkType = 1
	BPF_LINK_TYPE_TRACING        linkType = 2
	BPF_LINK_TYPE_CGROUP         linkType = 3
	BPF_LINK_TYPE_ITER           linkType = 4
	BPF_LINK_TYPE_NETNS          linkType = 5
	BPF_LINK_TYPE_XDP            linkType = 6
	BPF_LINK_TYPE_PERF_EVENT     linkType = 7
	BPF_LINK_TYPE_KPROBE_MULTI   linkType = 8
	BPF_LINK_TYPE_STRUCT_OPS     linkType = 9
	MAX_BPF_LINK_TYPE            linkType = 10
)

// linkId uniquely identifies a bpf_link.
type linkId uint32

type ebpflinkInfo struct {
	Type   linkType
	Id     linkId
	ProgId uint32
	_      [4]byte
	Extra  [16]uint8
}

var _ Info = (*ebpflinkInfo)(nil)

func (i *ebpflinkInfo) info() (unsafe.Pointer, uint32) {
	return unsafe.Pointer(i), uint32(unsafe.Sizeof(*i))
}

type LinkInformation struct {
	Type   linkType
	Id     int
	ProgId int
}

func LinkGetInfo(fd int) (LinkInformation, error) {
	var info ebpflinkInfo
	var information LinkInformation
	if err := objInfo(fd, &info); err != nil {
		return information, err
	}
	information.Type = info.Type
	information.Id = int(info.Id)
	information.ProgId = int(info.ProgId)

	return information, nil
}

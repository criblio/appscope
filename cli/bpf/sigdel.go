package bpf

import (
	"bytes"
	"encoding/binary"
	"errors"
	"os"

	"github.com/cilium/ebpf"
	"github.com/cilium/ebpf/perf"
)

type SigDelData struct {
	Pid     uint32
	NsPid   uint32
	Sig     uint32
	Errno   uint32
	Code    uint32
	Uid     uint32
	Gid     uint32
	Handler uint64
	Flags   uint64
	Comm    [32]byte
}

type SigDel struct {
	bpfLinkFd int
	bpfMap    *ebpf.Map
}

var (
	errSigDelCreateReader = errors.New("failed to create reader")
	errSigDelFailToRead   = errors.New("failed to read perf data")
)

// Terminate Sigdel
func (sd SigDel) Terminate() error {
	return sd.bpfMap.Close()
}

func (sd SigDel) ReadData(sigDelDataChan chan SigDelData) error {
	rd, err := perf.NewReader(sd.bpfMap, os.Getpagesize())
	if err != nil {
		return errSigDelCreateReader
	}
	defer rd.Close()
	for {
		ev, err := rd.Read()
		if err != nil {
			return errSigDelFailToRead
		}

		if ev.LostSamples != 0 {
			// log.Warn().Msgf("*** The perf event array buffer is full, dropped %d samples ****", ev.LostSamples)
			continue
		}

		b_arr := bytes.NewBuffer(ev.RawSample)

		var data SigDelData
		if err := binary.Read(b_arr, binary.LittleEndian, &data); err != nil {
			// log.Warn().Msgf("parsing perf event: %s", err)
			continue
		}

		sigDelDataChan <- data
	}
}

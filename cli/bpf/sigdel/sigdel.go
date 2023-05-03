//go:build exclude

package sigdel

import "C"

//go:generate go run github.com/cilium/ebpf/cmd/bpf2go -target bpfel -cc clang gen_sigdel ./sigdel.bpf.c -- -I/usr/include/bpf -I.

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"os"

	"github.com/rs/zerolog/log"

	"github.com/cilium/ebpf/link"
	"github.com/cilium/ebpf/perf"
	"golang.org/x/sys/unix"
)

type sigdel_data_t struct {
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

type SigEvent struct {
	sigdel_data_t
	CPU int
}

func setlimit() error {
	if err := unix.Setrlimit(unix.RLIMIT_MEMLOCK, &unix.Rlimit{
		Cur: unix.RLIM_INFINITY,
		Max: unix.RLIM_INFINITY,
	}); err != nil {
		log.Error().Err(err)
		return errors.New(fmt.Sprintf("failed to set temporary rlimit: %v", err))
	}
	return nil
}

func Sigdel(sigEventChan chan SigEvent) error {
	setlimit()

	objs := gen_sigdelObjects{}

	loadGen_sigdelObjects(&objs, nil)

	/*
	 * Trace events/types are described in /sys/kernel/debug/tracing/events
	 * This must be consistent with the .maps section name in C code.
	 */
	lnk, err := link.Tracepoint("signal", "signal_deliver", objs.SigDeliver, nil)
	if err != nil {
		log.Error().Msgf("*** ERROR: Set Tracepoint ***")
		return err
	}

	rd, err := perf.NewReader(objs.Events, os.Getpagesize())
	if err != nil {
		log.Error().Err(err)
		return errors.New("reader err")
	}

	defer objs.SigDeliver.Close()
	defer objs.Events.Close()
	defer lnk.Close()

	for {
		ev, err := rd.Read()
		if err != nil {
			log.Error().Err(err)
			return errors.New("Read fail error")
		}

		if ev.LostSamples != 0 {
			log.Warn().Msgf("*** The perf event array buffer is full, dropped %d samples ****", ev.LostSamples)
			continue
		}

		b_arr := bytes.NewBuffer(ev.RawSample)

		var data sigdel_data_t
		if err := binary.Read(b_arr, binary.LittleEndian, &data); err != nil {
			log.Warn().Msgf("parsing perf event: %s", err)
			continue
		}

		sigEventChan <- SigEvent{
			data,
			ev.CPU,
		}
	}
}

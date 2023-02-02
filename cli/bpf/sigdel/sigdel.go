package sigdel

import "C"
//go:generate go run github.com/cilium/ebpf/cmd/bpf2go -target bpfel -cc clang gen_sigdel ./sigdel.bpf.c -- -I/usr/include/bpf -I.

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"github.com/cilium/ebpf/link"
	"github.com/cilium/ebpf/perf"
	"golang.org/x/sys/unix"
	"log"
	"os"
)

type Sigdel_data_t struct {
	Pid     uint32
    Sig     uint32
    Errno   uint32
    Code    uint32
	Handler uint64
    Flags   uint64
    Comm   [32]byte
}

func setlimit() {
	if err := unix.Setrlimit(unix.RLIMIT_MEMLOCK, &unix.Rlimit {
    			Cur: unix.RLIM_INFINITY,
	            Max: unix.RLIM_INFINITY,
		}); err != nil {
		log.Fatalf("failed to set temporary rlimit: %v", err)
	}
}

func Sigdel() {
	setlimit()

	objs := gen_sigdelObjects{}

	loadGen_sigdelObjects(&objs, nil)
	link.Tracepoint("signal", "signal_deliver", objs.SigDeliver, nil)

	rd, err := perf.NewReader(objs.Events, os.Getpagesize())
	if err != nil {
		log.Fatalf("reader err")
	}

	for {
		ev, err := rd.Read()
		if err != nil {
			log.Fatalf("Read fail")
		}

		if ev.LostSamples != 0 {
			log.Printf("perf event ring buffer full, dropped %d samples", ev.LostSamples)
			continue
		}

		b_arr := bytes.NewBuffer(ev.RawSample)

		var data Sigdel_data_t
		if err := binary.Read(b_arr, binary.LittleEndian, &data); err != nil {
			log.Printf("parsing perf event: %s", err)
			continue
		}

		fmt.Printf("Signal CPU: %02d signal %d errno %d handler 0x%x pid: %d app %s\n",
        			ev.CPU, data.Sig, data.Errno, data.Handler,
                    data.Pid, data.Comm)
	}
}

package crash

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"time"

	"github.com/criblio/scope/internal"
	"github.com/criblio/scope/ipc"
	"github.com/criblio/scope/loader"
	"github.com/criblio/scope/util"
	"github.com/rs/zerolog/log"
	"github.com/shirou/gopsutil/v3/host"
	"github.com/shirou/gopsutil/v3/process"
)

type snapshot struct {
	// Source: self (mostly via eBPF)
	Time          time.Time // Time of Snapshot
	Version       string    // AppScope Cli Version
	SignalNumber  uint32    // Signal number
	SignalHandler string    // Signal handler
	Errno         uint32    // Error number
	ProcessName   string    // Process Name
	ProcessArgs   string    // Process Arguments
	Pid           uint32    // PID
	Uid           uint32    // User ID
	Gid           uint32    // Group ID

	// Source: /proc or linux
	Username    string   // Username  	// ebpf later?
	Environment []string // Environment Variables
	Arch        string   // Machine Arch
	Kernel      string   // Kernel version

	// Source: namespace
	Distro        string // Distro
	DistroVersion string // Distro version
	Hostname      string // Hostname

	// Maybe later:
	// JRE version (if java)
	// Go version (if go)
	// Static or Dynamically linked(?)
	// Application version(?)
	// Application and .so elf files
	// AppScope Log Output
	// scope ps output
	// scope history output
	// Container Impl (docker, podman...)
	// Container Name/Version?
	// SELinux or AppArmor enforcing?
	// Unix Capabilities... PTRACE?...
	// Namespace Id's
	// Network interface status?
	// ownership/permissions on pertinent files/unix sockets
	// dns on pertinent host names
}

// GenFiles creates the following files for a given pid
// - snapshot
// - info (where available)
// - coredump (where available)
// - cfg (where available)
// - backtrace (where available)
func GenFiles(sig, errno, pid, uid, gid uint32, sigHandler, procName, procArgs string) error {
	// TODO: If session directory exists, write to sessiondir/snapshot/
	// If not, write to /tmp/appscope/pid/
	dir := fmt.Sprintf("/tmp/appscope/%d", pid)
	if err := os.MkdirAll(dir, os.ModePerm); err != nil {
		log.Error().Err(err).Msgf("error creating snapshot directory")
		return err
	}

	// Where are we; Where is the pid?
	// Host ; Host
	// Host ; Container
	// Container ; This Container
	// Container ; Host
	// Container ; Another Container

	// Get pid of process inside namespace
	_, nsPid, err := ipc.IpcNsLastPidFromPid(ipc.IpcPidCtx{Pid: int(pid), PrefixPath: ""})
	if err != nil {
		log.Warn().Err(err).Msgf("Unable to get nspid for pid %d", pid)
	}

	ld := loader.New()

	// Retrieve info file if it doesn't exist
	infoFile := fmt.Sprintf("%s/info", dir)
	if !util.CheckFileExists(infoFile) {
		// Try to get from namespace
		infoFileNs := fmt.Sprintf("/tmp/appscope/%d/info", nsPid)
		_, err := ld.GetFile(infoFileNs, infoFile, int(pid))
		if err != nil {
			log.Warn().Err(err).Msgf("Unable to get %s file from namespace pid %d", infoFile, pid)
		}
	}

	// Retrieve coredump file if it doesn't exist
	coreFile := fmt.Sprintf("%s/core", dir)
	if !util.CheckFileExists(coreFile) {
		// Try to get from namespace
		coreFileNs := fmt.Sprintf("/tmp/appscope/%d/info", nsPid)
		_, err := ld.GetFile(coreFileNs, coreFile, int(pid))
		if err != nil {
			log.Warn().Err(err).Msgf("Unable to get %s file from namespace pid %d", coreFile, pid)
		}
	}

	// Retrieve cfg file if it doesn't exist
	cfgFile := fmt.Sprintf("%s/cfg", dir)
	if !util.CheckFileExists(cfgFile) {
		// Try to get from namespace
		cfgFileNs := fmt.Sprintf("/tmp/appscope/%d/info", nsPid)
		_, err := ld.GetFile(cfgFileNs, cfgFile, int(pid))
		if err != nil {
			log.Warn().Err(err).Msgf("Unable to get %s file from namespace pid %d", cfgFile, pid)
		}
	}

	// Retrieve backtrace file if it doesn't exist
	backtraceFile := fmt.Sprintf("%s/backtrace", dir)
	if !util.CheckFileExists(backtraceFile) {
		// Try to get from namespace
		backtraceFileNs := fmt.Sprintf("/tmp/appscope/%d/info", nsPid)
		_, err := ld.GetFile(backtraceFileNs, backtraceFile, int(pid))
		if err != nil {
			log.Warn().Err(err).Msgf("Unable to get %s file from namespace pid %d", backtraceFile, pid)
		}
	}

	// Create snapshot file
	snapshotFile := fmt.Sprintf("%s/snapshot", dir)
	if err := GenSnapshotFile(sig, errno, pid, uid, gid, sigHandler, procName, procArgs, snapshotFile); err != nil {
		log.Error().Err(err).Msgf("error generating snapshot file")
		return err
	}

	return nil
}

// GenSnapshotFile generates the snapshot file for a given pid
func GenSnapshotFile(sig, errno, pid, uid, gid uint32, sigHandler, procName, procArgs string, filePath string) error {
	var s snapshot

	// Source: self (mostly via eBPF)
	s.Time = time.Now()
	s.Version = internal.GetVersion()
	s.SignalNumber = sig
	s.SignalHandler = sigHandler
	s.Errno = errno
	s.ProcessName = procName
	s.ProcessArgs = procArgs
	s.Pid = pid
	s.Uid = uid
	s.Gid = gid

	// Source: /proc or linux
	p, err := process.NewProcess(int32(pid))
	if err != nil {
		log.Error().Err(err).Msgf("error getting process for pid %d", pid)
		return err
	}
	s.Username, err = p.Username()
	if err != nil {
		log.Error().Err(err).Msgf("error getting username for pid %d", pid)
		return err
	}
	s.Environment, err = p.Environ()
	if err != nil {
		log.Error().Err(err).Msgf("error getting environment for pid %d", pid)
		return err
	}
	s.Arch, err = host.KernelArch()
	if err != nil {
		log.Error().Err(err).Msgf("error getting kernel arch")
		return err
	}
	s.Kernel, err = host.KernelVersion()
	if err != nil {
		log.Error().Err(err).Msgf("error getting kernel version")
		return err
	}

	// Source: namespace
	// TODO: Distro, Distro version
	// TODO: Hostname

	// Create json structure
	jsonSnapshot, err := json.MarshalIndent(s, "", "  ")
	if err != nil {
		log.Error().Err(err).Msgf("error marshaling snapshot to json")
		return err
	}

	// Open and Write file to dir
	err = ioutil.WriteFile(filePath, jsonSnapshot, 0644)
	if err != nil {
		log.Error().Err(err).Msgf("error writing snapshot file")
		return err
	}

	return nil
}

// Resume sends a SIGCONT signal to the process allowing it to resume
func Resume(pid uint32) error {
	p, err := process.NewProcess(int32(pid))
	if err != nil {
		log.Error().Err(err).Msgf("error getting process for pid %d", pid)
		return err
	}

	return p.SendSignal(18)
}

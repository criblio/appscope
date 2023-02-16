package crash

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"strings"
	"syscall"
	"time"

	"github.com/criblio/scope/internal"
	"github.com/criblio/scope/ipc"
	"github.com/criblio/scope/util"
	"github.com/rs/zerolog/log"
	"github.com/shirou/gopsutil/v3/host"
	"github.com/shirou/gopsutil/v3/process"
)

type snapshot struct {
	// Source: self
	Time    time.Time
	Version string // AppScope Cli Version

	// Source: /proc or linux (or eBPF signal)
	Username      string   `json:",omitempty"`
	Environment   []string `json:",omitempty"`
	Arch          string   `json:",omitempty"`
	Kernel        string   `json:",omitempty"`
	SignalNumber  uint32   `json:",omitempty"`
	SignalHandler string   `json:",omitempty"`
	Errno         uint32   `json:",omitempty"`
	ProcessName   string   `json:",omitempty"`
	ProcessArgs   string   `json:",omitempty"`
	Pid           uint32
	Uid           uint32
	Gid           uint32

	// Source: namespace
	Hostname string `json:",omitempty"`

	// Maybe later:
	// Distro Version
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
func GenFiles(sig, errno, pid, uid, gid uint32, sigHandler uint64, procName, procArgs string) error {
	// TODO: If session directory exists, write to sessiondir/snapshot/
	// If not, write to /tmp/appscope/pid/
	dir := fmt.Sprintf("/tmp/appscope/%d", pid)
	syscall.Umask(0)
	if err := os.MkdirAll(dir, os.ModePerm); err != nil {
		log.Error().Err(err).Msgf("error creating snapshot directory")
		return err
	}

	// Is the pid provided in this container/this host?
	pidInThisNs, err := ipc.IpcNsIsSame(ipc.IpcPidCtx{Pid: int(pid), PrefixPath: ""})
	if err != nil {
		log.Error().Err(err).Msgf("error determining namespace")
		return err
	}

	// If process is in this container/this host (i.e. in this namespace) // proc/pid is visible
	//		(pid provided must be of this container/this host's perspective)
	//		crash files in place - do nothing here, just check they exist
	//		hostname file in place - do nothing here, just check it exists
	//		generate snapshot - call gensnapshot
	//			get process username,environ - from /proc
	if pidInThisNs {
		// File destinations and origins
		infoFile := fmt.Sprintf("%s/info", dir)
		coreFile := fmt.Sprintf("%s/core", dir)
		cfgFile := fmt.Sprintf("%s/cfg", dir)
		backtraceFile := fmt.Sprintf("%s/backtrace", dir)
		hostnameFile := fmt.Sprintf("/etc/hostname")
		snapshotFile := fmt.Sprintf("%s/snapshot", dir)

		// Info file
		if !util.CheckFileExists(infoFile) {
			log.Warn().Err(err).Msgf("Unable to find %s file for pid %d", infoFile, pid)
		}

		// Coredump file
		if !util.CheckFileExists(coreFile) {
			log.Warn().Err(err).Msgf("Unable to find %s file for pid %d", coreFile, pid)
		}

		// Cfg file
		if !util.CheckFileExists(cfgFile) {
			log.Warn().Err(err).Msgf("Unable to find %s file for pid %d", cfgFile, pid)
		}

		// Backtrace file
		if !util.CheckFileExists(backtraceFile) {
			log.Warn().Err(err).Msgf("Unable to find %s file for pid %d", backtraceFile, pid)
		}

		// Hostname file
		if !util.CheckFileExists(hostnameFile) {
			log.Warn().Err(err).Msgf("Unable to find %s file for pid %d", hostnameFile, pid)
		}

		// Snapshot file
		if err := GenSnapshotFile(sig, errno, pid, uid, gid, sigHandler, procName, procArgs, hostnameFile, snapshotFile); err != nil {
			log.Error().Err(err).Msgf("error generating snapshot file")
			return err
		}
	}

	// If process is in a container below us (i.e. in a below namespace) // proc/pid is visible
	// 		(pid provided must be of this container/this host's perspective)
	// 		get crash files - use /proc/pid/root
	// 		get hostname file - use /proc/pid/root
	// 		generate snapshot - call gensnapshot
	// 			get process username,environ - from /proc
	if !pidInThisNs {
		// File destinations
		infoFile := fmt.Sprintf("%s/info", dir)
		coreFile := fmt.Sprintf("%s/core", dir)
		cfgFile := fmt.Sprintf("%s/cfg", dir)
		backtraceFile := fmt.Sprintf("%s/backtrace", dir)
		hostnameFile := fmt.Sprintf("/%s/hostname", dir)
		snapshotFile := fmt.Sprintf("%s/snapshot", dir)

		// Get pid of process inside namespace
		_, nsPid, err := ipc.IpcNsLastPidFromPid(ipc.IpcPidCtx{Pid: int(pid), PrefixPath: ""})
		if err != nil {
			log.Warn().Err(err).Msgf("Unable to get nspid for pid %d", pid)
		}

		// File origins
		nsDir := fmt.Sprintf("/proc/%d/root/tmp/appscope/%d", pid, nsPid)
		nsInfoFile := fmt.Sprintf("%s/info", nsDir)
		nsCoreFile := fmt.Sprintf("%s/core", nsDir)
		nsCfgFile := fmt.Sprintf("%s/cfg", nsDir)
		nsBacktraceFile := fmt.Sprintf("%s/backtrace", nsDir)
		nsHostnameFile := fmt.Sprintf("/proc/%d/root/etc/hostname", pid)

		// Info file
		if _, err = util.CopyFile2(nsInfoFile, infoFile); err != nil {
			log.Warn().Err(err).Msgf("Unable to get %s file from namespace for pid %d", nsInfoFile, pid)
		}

		// Coredump file
		if _, err = util.CopyFile2(nsCoreFile, coreFile); err != nil {
			log.Warn().Err(err).Msgf("Unable to get %s file from namespace for pid %d", nsCoreFile, pid)
		}

		// Cfg file
		if _, err = util.CopyFile2(nsCfgFile, cfgFile); err != nil {
			log.Warn().Err(err).Msgf("Unable to get %s file from namespace for pid %d", nsCfgFile, pid)
		}

		// Backtrace file
		if _, err = util.CopyFile2(nsBacktraceFile, backtraceFile); err != nil {
			log.Warn().Err(err).Msgf("Unable to get %s file from namespace for pid %d", nsBacktraceFile, pid)
		}

		// Hostname file
		if _, err = util.CopyFile2(nsHostnameFile, hostnameFile); err != nil {
			log.Warn().Err(err).Msgf("Unable to get %s file from namespace for pid %d", nsHostnameFile, pid)
		}

		// Snapshot file
		if err := GenSnapshotFile(sig, errno, pid, uid, gid, sigHandler, procName, procArgs, hostnameFile, snapshotFile); err != nil {
			log.Error().Err(err).Msgf("error generating snapshot file")
			return err
		}

		// TODO delete hostname file
	}

	// If process is in a parallel container or a host above (i.e. in an above / parallel namespace) // proc/pid not visible unless hostfs
	//		(pid provided must be of the hosts perspective)
	//		(requires --privileged flag? or mounted hostfs?)
	//		get crash files - use /proc/pid/root or --getfiles
	//		get hostname file - use /proc/pid/root or --getfiles
	//		generate snapshot - call gensnapshot
	//			get process username,environ - from /proc using --getfiles

	// Unsupported at this time.

	return nil
}

// GenSnapshotFile generates the snapshot file for a given pid
func GenSnapshotFile(sig, errno, pid, uid, gid uint32, sigHandler uint64, procName, procArgs, hostnameFilePath, filePath string) error {
	var s snapshot

	// Source: self
	s.Time = time.Now()
	s.Version = internal.GetVersion()
	s.Pid = pid

	// Source: /proc or linux (or eBPF signal)
	var err error
	s.Arch, err = host.KernelArch()
	if err != nil {
		log.Warn().Err(err).Msgf("unable to get kernel arch")
	}
	s.Kernel, err = host.KernelVersion()
	if err != nil {
		log.Warn().Err(err).Msgf("unable to get kernel version")
	}
	p, err := process.NewProcess(int32(pid))
	if err != nil {
		log.Error().Err(err).Msgf("error getting process for pid %d", pid)
		return err
	}
	s.Username, err = p.Username()
	if err != nil {
		log.Warn().Err(err).Msgf("unable to get username for pid %d", pid)
	}
	s.Environment, err = p.Environ()
	if err != nil {
		log.Warn().Err(err).Msgf("unable to get environment for pid %d", pid)
	}
	if sig == 0 { // Initiated by snapshot command, need to get these values ourselves
		uids, err := p.Uids()
		if err != nil {
			log.Warn().Err(err).Msgf("unable to get uid for pid %d", pid)
		}
		s.Uid = uint32(uids[0])
		gids, err := p.Gids()
		if err != nil {
			log.Warn().Err(err).Msgf("unable to get gid for pid %d", pid)
		}
		s.Gid = uint32(gids[0])
		s.ProcessName, err = p.Name()
		if err != nil {
			log.Warn().Err(err).Msgf("unable to get name for pid %d", pid)
		}
		s.ProcessArgs, err = p.Cmdline()
		if err != nil {
			log.Warn().Err(err).Msgf("unable to get args for pid %d", pid)
		}
	} else { // Initiated by signal, we know these values
		s.SignalNumber = sig
		s.SignalHandler = fmt.Sprintf("0x%x", sigHandler)
		s.Errno = errno
		s.Uid = uid
		s.Gid = gid
		s.ProcessName = strings.Trim(procName, "\x00")
		s.ProcessArgs = procArgs
	}

	// Source: namespace
	hfBytes, err := ioutil.ReadFile(hostnameFilePath)
	if err == nil {
		s.Hostname = strings.TrimSpace(string(hfBytes))
	}

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

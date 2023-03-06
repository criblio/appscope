package snapshot

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
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

type oomCrashInfo struct {
	// Source: self
	Time    time.Time
	Version string
	// Source: eBPF
	Pid         uint32
	ProcessName string `json:",omitempty"`
}
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
	snapshotFile := fmt.Sprintf("%s/snapshot_%d", dir, time.Now().Unix())

	// If process is in this container/this host (i.e. in this namespace) // proc/pid is visible
	//		(pid provided must be of this container/this host's perspective)
	//		crash files in place - do nothing here, just check they exist
	//		hostname file in place - do nothing here, just check it exists
	//		generate snapshot - call gensnapshot
	//			get process username,environ - from /proc
	if pidInThisNs {
		// Host Name and snapshot
		hostnameFile := "/etc/hostname"

		files, err := os.ReadDir(dir)
		if err != nil {
			log.Warn().Err(err).Msgf("Unable to read %s directory for pid %d", dir, pid)
		}

		var libSnapFilesPrefixes = map[string]bool{
			"info_":      false,
			"core_":      false,
			"cfg_":       false,
			"backtrace_": false,
		}

		// Iterate over files
		for _, file := range files {
			if !file.Type().IsRegular() {
				continue
			}

			for libSnapPrefix := range libSnapFilesPrefixes {
				if strings.HasPrefix(file.Name(), libSnapPrefix) {
					libSnapFilesPrefixes[libSnapPrefix] = true
				}
			}
		}

		for libSnapPrefix, val := range libSnapFilesPrefixes {
			if !val {
				log.Warn().Msgf("Unable to find %s file for pid %d", filepath.Join(dir, libSnapPrefix), pid)
			}
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
	} else {
		// If process is in a container below us (i.e. in a below namespace) // proc/pid is visible
		// 		(pid provided must be of this container/this host's perspective)
		// 		get crash files - use /proc/pid/root
		// 		get hostname file - use /proc/pid/root
		// 		generate snapshot - call gensnapshot
		// 			get process username,environ - from /proc
		// Host Name and snapshot
		hostnameFile := fmt.Sprintf("/%s/hostname", dir)

		// Get pid of process inside namespace
		_, nsPid, err := ipc.IpcNsLastPidFromPid(ipc.IpcPidCtx{Pid: int(pid), PrefixPath: ""})
		if err != nil {
			log.Warn().Err(err).Msgf("Unable to get nspid for pid %d", pid)
		}

		// File origins
		nsDir := fmt.Sprintf("/proc/%d/root/tmp/appscope/%d", pid, nsPid)
		nsHostnameFile := fmt.Sprintf("/proc/%d/root/etc/hostname", pid)

		files, err := os.ReadDir(nsDir)
		if err != nil {
			log.Warn().Err(err).Msgf("Unable to read %s namespace directory for pid %d", nsDir, pid)
		}

		libSnapFilesPrefixes := []string{"info_", "core_", "cfg_", "backtrace_"}

		// Iterate over files
		for _, file := range files {
			if !file.Type().IsRegular() {
				continue
			}

			for _, libSnapPrefix := range libSnapFilesPrefixes {
				fileName := file.Name()
				if strings.HasPrefix(fileName, libSnapPrefix) {
					libSnapSrcFile := filepath.Join(nsDir, fileName)
					libSnapDestFile := filepath.Join(dir, fileName)
					if _, err = util.CopyFile2(libSnapSrcFile, libSnapDestFile); err != nil {
						log.Warn().Err(err).Msgf("Unable to get %s file from namespace for pid %d", libSnapSrcFile, pid)
					}
				}
			}
		}

		// Hostname file
		if _, err = util.CopyFile2(nsHostnameFile, hostnameFile); err != nil {
			log.Warn().Err(err).Msgf("Unable to get %s file from namespace for pid %d", nsHostnameFile, pid)
		}
		defer os.Remove(hostnameFile)

		// Snapshot file
		if err := GenSnapshotFile(sig, errno, pid, uid, gid, sigHandler, procName, procArgs, hostnameFile, snapshotFile); err != nil {
			log.Error().Err(err).Msgf("error generating snapshot file")
			return err
		}
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
	env, err := p.Environ()
	if err != nil {
		log.Warn().Err(err).Msgf("unable to get environment for pid %d", pid)
	}
	s.Environment = util.RemoveEmptyStrings(env)
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
	hfBytes, err := os.ReadFile(hostnameFilePath)
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
	err = os.WriteFile(filePath, jsonSnapshot, 0644)
	if err != nil {
		log.Error().Err(err).Msgf("error writing snapshot file")
		return err
	}

	return nil
}

// GenSnapshotOOmFile generates the OOM file for a given pid
func GenSnapshotOOmFile(pid uint32, procName, filepath string) error {
	var ooms oomCrashInfo

	ooms.Time = time.Now()
	ooms.Version = internal.GetVersion()
	ooms.Pid = pid
	ooms.ProcessName = strings.Trim(procName, "\x00")

	// Create json structure
	jsonOomSnapshot, err := json.MarshalIndent(ooms, "", "  ")
	if err != nil {
		log.Error().Err(err).Msgf("error marshaling OOM snapshot to json")
		return err
	}

	// Open and Write file to dir
	err = os.WriteFile(filepath, jsonOomSnapshot, 0644)
	if err != nil {
		log.Error().Err(err).Msgf("error writing OOM snapshot file")
		return err
	}
	return nil
}

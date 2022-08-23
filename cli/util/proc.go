package util

import (
	"errors"
	"fmt"
	"io/ioutil"
	"os"
	"os/user"
	"strconv"
	"strings"

	linuxproc "github.com/c9s/goprocinfo/linux"
)

// Process is a unix process
type Process struct {
	ID      int    `json:"id"`
	Pid     int    `json:"pid"`
	User    string `json:"user"`
	Scoped  bool   `json:"scoped"`
	Command string `json:"command"`
}

// Processes is an array of Process
type Processes []Process

var (
	ErrOpenProc       = errors.New("cannot open proc directory")
	ErrReadProc       = errors.New("cannot read from proc directory")
	ErrGetProcStatus  = errors.New("error getting process status")
	ErrGetProcCmdLine = errors.New("error getting process command line")
	ErrMissingUser    = errors.New("unable to find user")
)

// ProcessesByName returns an array of processes that match a given name
func ProcessesByName(name string) (Processes, error) {
	processes := make([]Process, 0)

	procDir, err := os.Open("/proc")
	if err != nil {
		return processes, ErrOpenProc
	}
	defer procDir.Close()

	procs, err := procDir.Readdirnames(0)
	if err != nil {
		return processes, ErrReadProc
	}

	i := 1
	for _, p := range procs {
		// Skip non-integers as they are not PIDs
		if !IsNumeric(p) {
			continue
		}

		// Convert directory name to integer
		pid, err := strconv.Atoi(p)
		if err != nil {
			continue
		}

		command, err := PidCommand(pid)
		if err != nil {
			continue
		}

		cmdLine, err := PidCmdline(pid)
		if err != nil {
			continue
		}

		// TODO in container namespace we cannot depend on following info
		userName, err := PidUser(pid)
		if err != nil && !errors.Is(err, ErrMissingUser) {
			continue
		}

		if strings.Contains(command, name) {
			processes = append(processes, Process{
				ID:      i,
				Pid:     pid,
				User:    userName,
				Scoped:  PidScoped(pid),
				Command: cmdLine,
			})
			i++
		}
	}
	return processes, nil
}

// ProcessesScoped returns an array of processes that are currently being scoped
func ProcessesScoped() (Processes, error) {
	processes := make([]Process, 0)

	procDir, err := os.Open("/proc")
	if err != nil {
		return processes, ErrOpenProc
	}
	defer procDir.Close()

	procs, err := procDir.Readdirnames(0)
	if err != nil {
		return processes, ErrReadProc
	}

	i := 1
	for _, p := range procs {
		// Skip non-integers as they are not PIDs
		if !IsNumeric(p) {
			continue
		}

		// Convert directory name to integer
		pid, err := strconv.Atoi(p)
		if err != nil {
			continue
		}

		cmdLine, err := PidCmdline(pid)
		if err != nil {
			continue
		}

		// TODO in container namespace we cannot depend on following info
		userName, err := PidUser(pid)
		if err != nil && !errors.Is(err, ErrMissingUser) {
			continue
		}

		// Add process if is is scoped
		scoped := PidScoped(pid)
		if scoped {
			processes = append(processes, Process{
				ID:      i,
				Pid:     pid,
				User:    userName,
				Scoped:  scoped,
				Command: cmdLine,
			})
			i++
		}
	}
	return processes, nil
}

// PidUser returns the user owning the process specified by PID
func PidUser(pid int) (string, error) {

	// Get uid from status
	pStat, err := linuxproc.ReadProcessStatus(fmt.Sprintf("/proc/%v/status", pid))
	if err != nil {
		return "", ErrGetProcStatus
	}

	// Lookup username by uid
	user, err := user.LookupId(fmt.Sprint(pStat.RealUid))
	if err != nil {
		return "", ErrMissingUser
	}

	return user.Username, nil
}

// PidScoped checks if a process specified by PID is being scoped
func PidScoped(pid int) bool {
	pidMapFile, err := ioutil.ReadFile(fmt.Sprintf("/proc/%v/maps", pid))
	if err != nil {
		return false
	}

	// Look for libscope in map
	pidMap := string(pidMapFile)
	if strings.Contains(pidMap, "libscope") {
		command, err := PidCommand(pid)
		if err != nil {
			return false
		}
		return command != "ldscopedyn"
	}

	return false
}

// PidCommand gets the command used to start the process specified by PID
func PidCommand(pid int) (string, error) {
	// Get command from status
	pStat, err := linuxproc.ReadProcessStatus(fmt.Sprintf("/proc/%v/status", pid))
	if err != nil {
		return "", ErrGetProcStatus
	}

	return pStat.Name, nil
}

// PidCmdline gets the cmdline used to start the process specified by PID
func PidCmdline(pid int) (string, error) {
	// Get cmdline
	cmdline, err := linuxproc.ReadProcessCmdline(fmt.Sprintf("/proc/%v/cmdline", pid))
	if err != nil {
		return "", ErrGetProcCmdLine

	}

	return cmdline, nil
}

// PidExists checks if a PID is valid
func PidExists(pid int) bool {
	pidPath := fmt.Sprintf("/proc/%v", pid)
	if CheckDirExists(pidPath) {
		return true
	}
	return false
}

func PidService(pid int) (bool, error) {
	// Get ppid from status
	pStat, err := linuxproc.ReadProcessStatus(fmt.Sprintf("/proc/%v/status", pid))
	if err != nil {
		return false, ErrGetProcStatus
	}
	return pStat.PPid == 1, nil
}

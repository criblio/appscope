package util

import (
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
	Command string `json:"command"`
	Scoped  bool   `json:"scoped"`
}

// Processes is an array of Process
type Processes []Process

// ProcessesByName returns an array of processes that match a given name
func ProcessesByName(name string) Processes {
	processes := make([]Process, 0)

	procDir, err := os.Open("/proc")
	if err != nil {
		ErrAndExit("Cannot open proc directory")
	}
	defer procDir.Close()

	procs, err := procDir.Readdirnames(0)
	if err != nil {
		ErrAndExit("Cannot read from proc directory")
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
			ErrAndExit("Error converting process name to integer: %s", err)
		}

		// Add process if there is a name match
		command := PidCommand(pid)
		if strings.Contains(command, name) {
			processes = append(processes, Process{
				ID:      i,
				Pid:     pid,
				User:    PidUser(pid),
				Command: command,
				Scoped:  PidScoped(pid),
			})
			i++
		}
	}
	return processes
}

// ProcessesScoped returns an array of processes that are currently being scoped
func ProcessesScoped() Processes {
	processes := make([]Process, 0)

	procDir, err := os.Open("/proc")
	if err != nil {
		ErrAndExit("Cannot open proc directory")
	}
	defer procDir.Close()

	procs, err := procDir.Readdirnames(0)
	if err != nil {
		ErrAndExit("Cannot read from proc directory")
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
			ErrAndExit("Error converting process name to integer: %s", err)
		}

		// Add process if is is scoped
		scoped := PidScoped(pid)
		if scoped {
			processes = append(processes, Process{
				ID:      i,
				Pid:     pid,
				User:    PidUser(pid),
				Command: PidCommand(pid),
				Scoped:  scoped,
			})
			i++
		}
	}
	return processes
}

// PidUser returns the user owning the process specified by PID
func PidUser(pid int) string {
	pidPath := fmt.Sprintf("/proc/%v", pid)

	// Get uid from status
	pStat, err := linuxproc.ReadProcessStatus(pidPath + "/status")
	if err != nil {
		ErrAndExit("Error getting uid: %v", err)
	}

	// Lookup username by uid
	user, err := user.LookupId(fmt.Sprint(pStat.RealUid))
	if err != nil {
		ErrAndExit("Unable to find user: %v", err)
	}

	return user.Username
}

// PidScoped checks if a process specified by PID is being scoped
func PidScoped(pid int) bool {
	pidPath := fmt.Sprintf("/proc/%v", pid)
	pidMapFile, err := ioutil.ReadFile(pidPath + "/maps")
	if err != nil {
		return false
	}

	// Look for libscope in map
	pidMap := string(pidMapFile)
	if strings.Contains(pidMap, "libscope") {
		if PidCommand(pid) != "ldscopedyn" {
			return true
		}
	}

	return false
}

// PidCommand gets the command used to start the process specified by PID
func PidCommand(pid int) string {
	pidPath := fmt.Sprintf("/proc/%v", pid)

	// Get name from status
	pStat, err := linuxproc.ReadProcessStatus(pidPath + "/status")
	if err != nil {
		ErrAndExit("Error getting process name: %v", err)
	}

	return pStat.Name
}

// PidExists checks if a PID is valid
func PidExists(pid int) bool {
	pidPath := fmt.Sprintf("/proc/%v", pid)
	if CheckDirExists(pidPath) {
		return true
	}
	return false
}

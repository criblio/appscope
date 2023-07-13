package util

import (
	"bufio"
	"errors"
	"fmt"
	"os"
	"os/user"
	"strconv"
	"strings"

	linuxproc "github.com/c9s/goprocinfo/linux"
	"github.com/criblio/scope/ipc"
)

// Process is a unix process
type Process struct {
	ID      int    `json:"id"`
	Pid     int    `json:"pid"`
	User    string `json:"user"`
	Scoped  bool   `json:"scoped"`
	Command string `json:"command"`
}

// PidScopeMap is a map of Pid and Scope state
type PidScopeMapState map[int]bool

// Processes is an array of Process
type Processes []Process

var (
	errOpenProc         = errors.New("cannot open proc directory")
	errGetProcStatus    = errors.New("error getting process status")
	errGetProcCmdLine   = errors.New("error getting process command line")
	errGetProcTask      = errors.New("error getting process task")
	errGetProcChildren  = errors.New("error getting process children")
	errPidMissing       = errors.New("error pid does not exist")
	errMissingUser      = errors.New("unable to find user")
	errPidInvalid       = errors.New("invalid PID")
	errInvalidSelection = errors.New("invalid Selection")
)

// searchPidByProcName check if specified inputProcName fully match the pid's process name
func searchPidByProcName(rootdir string, pid int, inputProcName string) bool {

	procName, err := PidCommand(rootdir, pid)
	if err != nil {
		return false
	}
	return inputProcName == procName
}

// searchPidByCmdLine check if specified inputArg submatch the pid's command line
func searchPidByCmdLine(rootdir string, pid int, inputArg string) bool {

	cmdLine, err := PidCmdline(rootdir, pid)
	if err != nil {
		return false
	}
	return strings.Contains(cmdLine, inputArg)
}

type searchFunc func(string, int, string) bool

// pidScopeMapSearch returns an map of processes that met conditions in searchFunc
func pidScopeMapSearch(rootdir, inputArg string, sF searchFunc) (PidScopeMapState, error) {
	pidMap := make(PidScopeMapState)

	procs, err := pidProcDirsNames(rootdir)
	if err != nil {
		return pidMap, err
	}

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

		if sF(rootdir, pid, inputArg) {
			status, err := PidScopeStatus(rootdir, pid)
			if err != nil {
				continue
			}
			pidMap[pid] = (status == Active)
		}
	}

	return pidMap, nil
}

// PidScopeMapByProcessName returns an map of processes name that are found by process name match
func PidScopeMapByProcessName(rootdir, procname string) (PidScopeMapState, error) {
	return pidScopeMapSearch(rootdir, procname, searchPidByProcName)
}

// PidScopeMapByCmdLine returns an map of processes that are found by cmdLine submatch
func PidScopeMapByCmdLine(rootdir, cmdLine string) (PidScopeMapState, error) {
	return pidScopeMapSearch(rootdir, cmdLine, searchPidByCmdLine)
}

// pidProcDirsNames returns a list with process directory names
func pidProcDirsNames(rootdir string) ([]string, error) {
	procPath := fmt.Sprintf("%s/proc", rootdir)
	procDir, err := os.Open(procPath)
	if err != nil {
		return nil, errOpenProc
	}
	defer procDir.Close()

	return procDir.Readdirnames(0)
}

// Returns all processes that match the name, only if scope is attached
// ProcessesByNameToDetach returns an array of processes to detach that match a given name
func ProcessesByNameToDetach(rootdir, name, procArg string, exactMatch bool) (Processes, error) {
	return processesByName(rootdir, name, procArg, true, exactMatch)
}

// Returns all processes that match the name, regardless of their state (attached, detached, loaded, unloaded)
// ProcessesByNameToAttach returns an array of processes to attach that match a given name
func ProcessesByNameToAttach(rootdir, name, procArg string, exactMatch bool) (Processes, error) {
	return processesByName(rootdir, name, procArg, false, exactMatch)
}

// processesByName returns an array of processes that match a given name
func processesByName(rootdir, name, procArg string, activeOnly, exactMatch bool) (Processes, error) {
	processes := make([]Process, 0)

	procs, err := pidProcDirsNames(rootdir)
	if err != nil {
		return processes, err
	}

	i := 1
	for _, p := range procs {
		// Skip non-integers as they are not PIDs
		if !IsNumeric(p) {
			continue
		}

		// Skip if no permission to read the fd directory
		filePath := fmt.Sprintf("%s/proc/%v/fd", rootdir, p)
		procFdDir, err := os.Open(filePath)
		if err != nil {
			continue
		}
		procFdDir.Close()

		// Convert directory name to integer
		pid, err := strconv.Atoi(p)
		if err != nil {
			continue
		}

		command, err := PidCommand(rootdir, pid)
		if err != nil {
			continue
		}

		cmdLine, err := PidCmdline(rootdir, pid)
		if err != nil {
			continue
		}

		// TODO in container namespace we cannot depend on following info
		userName, err := PidUser(rootdir, pid)
		if err != nil && !errors.Is(err, errMissingUser) {
			continue
		}

		status, err := PidScopeStatus(rootdir, pid)
		if err != nil {
			continue
		}

		// Add process if there is a name match
		if (exactMatch && command == name) || (!exactMatch && strings.Contains(command, name)) {
			if strings.Contains(cmdLine, procArg) {
				if !activeOnly || (activeOnly && status == Active) {
					processes = append(processes, Process{
						ID:      i,
						Pid:     pid,
						User:    userName,
						Scoped:  status == Active,
						Command: cmdLine,
					})
					i++
				}
			}
		}
	}
	return processes, nil
}

// ProcessesScoped returns an array of processes that are currently being scoped
func ProcessesScoped(rootdir string) (Processes, error) {
	processes := make([]Process, 0)

	procs, err := pidProcDirsNames(rootdir)
	if err != nil {
		return processes, err
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

		cmdLine, err := PidCmdline(rootdir, pid)
		if err != nil {
			continue
		}

		userName, err := PidUser(rootdir, pid)
		if err != nil && !errors.Is(err, errMissingUser) {
			continue
		}

		// Add process if is is scoped
		status, err := PidScopeStatus(rootdir, pid)
		if err != nil {
			continue
		}
		if status == Active {
			processes = append(processes, Process{
				ID:      i,
				Pid:     pid,
				User:    userName,
				Scoped:  true,
				Command: cmdLine,
			})
			i++
		}
	}
	return processes, nil
}

// ProcessesToDetach returns an array of processes that can be detached
func ProcessesToDetach(rootdir string) (Processes, error) {
	processes := make([]Process, 0)

	procs, err := pidProcDirsNames(rootdir)
	if err != nil {
		return processes, err
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

		cmdLine, err := PidCmdline(rootdir, pid)
		if err != nil {
			continue
		}

		userName, err := PidUser(rootdir, pid)
		if err != nil && !errors.Is(err, errMissingUser) {
			continue
		}

		// Detach the process in case the situation we are not able to retrieve the info
		status, err := PidScopeStatus(rootdir, pid)
		if err != nil && !errors.Is(err, ipc.ErrConsumerTimeout) {
			continue
		}
		// Add process if is is actively scoped
		if status == Active || errors.Is(err, ipc.ErrConsumerTimeout) {
			processes = append(processes, Process{
				ID:      i,
				Pid:     pid,
				User:    userName,
				Scoped:  true,
				Command: cmdLine,
			})
			i++
		}
	}
	return processes, nil
}

// PidUser returns the user owning the process specified by PID
func PidUser(rootdir string, pid int) (string, error) {

	// Get uid from status
	pStat, err := linuxproc.ReadProcessStatus(fmt.Sprintf("%s/proc/%v/status", rootdir, pid))
	if err != nil {
		return "", errGetProcStatus
	}

	// TODO add support for foreign container
	// Lookup username by uid
	user, err := user.LookupId(fmt.Sprint(pStat.RealUid))
	if err != nil {
		return "", errMissingUser
	}

	return user.Username, nil
}

// PidScopeLibInMaps checks if a process specified by PID contains libscope in memory mappings.
func PidScopeLibInMaps(rootdir string, pid int) (bool, error) {
	pidMapFile, err := os.ReadFile(fmt.Sprintf("%s/proc/%v/maps", rootdir, pid))
	if err != nil {
		// Process or do not exist or we do not have permissions to read a map file
		return false, err
	}

	pidMap := string(pidMapFile)
	return strings.Contains(pidMap, "libscope"), nil
}

// PidScopeStatus checks a Scope Status if a process specified by PID.
func PidScopeStatus(rootdir string, pid int) (ScopeStatus, error) {
	if !PidExists(rootdir, pid) {
		return Disable, errPidMissing
	}

	pidMapFile, err := os.ReadFile(fmt.Sprintf("%s/proc/%v/maps", rootdir, pid))
	if err != nil {
		// Process or do not exist or we do not have permissions to read a map file
		return Disable, err
	}

	pidMap := string(pidMapFile)
	if !strings.Contains(pidMap, "libscope") {
		// Process does not contain libscope library in maps
		return Disable, nil
	}

	// Ignore scopedyn processes
	// TODO: Still intended?
	command, err := PidCommand(rootdir, pid)
	if err != nil {
		return Disable, nil
	}
	if command == "scopedyn" {
		return Disable, nil
	}

	// Check shmem does not exist (if scope_anon does not exist the proc is scoped)
	files, err := os.ReadDir(fmt.Sprintf("%s/proc/%v/fd", rootdir, pid))
	if err != nil {
		// Process or do not exist or we do not have permissions to read a fd file
		return Disable, err
	}

	for _, file := range files {
		filePath := fmt.Sprintf("%s/proc/%v/fd/%s", rootdir, pid, file.Name())
		resolvedFileName, err := os.Readlink(filePath)
		if err != nil {
			continue
		}
		if strings.Contains(resolvedFileName, "scope_anon") {
			return Setup, nil
		}
	}

	// Retrieve information from IPC
	return getScopeStatus(rootdir, pid)
}

// PidCommand gets the command used to start the process specified by PID
func PidCommand(rootdir string, pid int) (string, error) {
	// Get command from status
	pStat, err := linuxproc.ReadProcessStatus(fmt.Sprintf("%s/proc/%v/status", rootdir, pid))
	if err != nil {
		return "", errGetProcStatus
	}

	return pStat.Name, nil
}

// PidCmdline gets the cmdline used to start the process specified by PID
func PidCmdline(rootdir string, pid int) (string, error) {
	// Get cmdline
	cmdline, err := linuxproc.ReadProcessCmdline(fmt.Sprintf("%s/proc/%v/cmdline", rootdir, pid))
	if err != nil {
		return "", errGetProcCmdLine
	}

	return cmdline, nil
}

// PidInitContainer verify if specific PID is the init PID in the container
func PidInitContainer(rootdir string, pid int) (bool, error) {
	// TODO: goprocinfo does not support all the status parameters (NsPid)
	// handle procfs by ourselves ?
	file, err := os.Open(fmt.Sprintf("%s/proc/%v/status", rootdir, pid))
	if err != nil {
		return false, errGetProcStatus
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)
	scanner.Split(bufio.ScanLines)
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "NSpid:") {
			// Skip Nspid
			nsPidString := strings.Fields(line)[1:]
			// Check for nested PID namespace and the init PID in namespace (it should be equals 1)
			if (len(nsPidString) > 1) && (nsPidString[len(nsPidString)-1] == "1") {
				return true, nil
			}
		}
	}
	return false, nil
}

// PidChildren retrieves the children PID's for the main process specified by the PID
func PidChildren(rootdir string, pid int) ([]int, error) {
	file, err := os.Open(fmt.Sprintf("%s/proc/%v/task/%v/children", rootdir, pid, pid))
	if err != nil {
		return nil, errGetProcChildren
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	scanner.Split(bufio.ScanWords)
	var childrenPids []int
	for scanner.Scan() {
		childPid, err := strconv.Atoi(scanner.Text())
		if err != nil {
			return nil, errGetProcChildren
		}
		childrenPids = append(childrenPids, childPid)
	}
	return childrenPids, nil
}

// PidThreadsPids gets the all the thread PIDs specified by PID
func PidThreadsPids(rootdir string, pid int) ([]int, error) {
	files, err := os.ReadDir(fmt.Sprintf("%s/proc/%v/task", rootdir, pid))
	if err != nil {
		return nil, errGetProcTask
	}

	threadPids := make([]int, len(files))

	for _, file := range files {
		tid, _ := strconv.Atoi(file.Name())
		threadPids = append(threadPids, tid)
	}

	return threadPids, nil
}

// PidExists checks if a PID is valid
func PidExists(rootdir string, pid int) bool {
	pidPath := fmt.Sprintf("%s/proc/%v", rootdir, pid)
	return CheckDirExists(pidPath)
}

// containerPids returns list of PID's of currently running containers
func containerPids(rootdir string) []int {
	cPids := []int{}
	ctrFuncs := []func(string) ([]int, error){GetContainerDPids, GetPodmanPids, GetLXCPids}

	for _, ctrFunc := range ctrFuncs {
		ctrPids, err := ctrFunc(rootdir)
		if err != nil {
			continue
		}
		cPids = append(cPids, ctrPids...)
	}

	return cPids
}

// PidGetRefPidForMntNamespace returns reference PID of mnt namespace,
// Returns -1 if the refrence PID is the same as the scope client PID
func PidGetRefPidForMntNamespace(rootdir string, targetPid int) int {
	targetInfo, err := os.Readlink(fmt.Sprintf("%s/proc/%d/ns/mnt", rootdir, targetPid))
	if err != nil {
		return -1
	}

	// First check if the namespace used by process is the same namespace as CLI
	nsInfo, err := os.Readlink(fmt.Sprintf("%s/proc/self/ns/mnt", rootdir))
	if err != nil {
		return -1
	}

	if nsInfo == targetInfo {
		return -1
	}

	// Check if the namespace used by process belongs to one of the detected containers
	ctrPids := containerPids(rootdir)
	for _, nsPid := range ctrPids {
		nsInfo, err := os.Readlink(fmt.Sprintf("%s/proc/%d/ns/mnt", rootdir, nsPid))
		if err != nil {
			continue
		}

		if nsInfo == targetInfo {
			return nsPid
		}
	}

	// Assume that target process do not exists in the container but have seperated mount namespace
	return targetPid
}

// HandleInputArg handles the input argument (process name or pid) and returns an array of processes
// If the id provided is a pid, that single pid will be returned as a proc
// If the id provided is a name, all processes matching that name will be returned in procs
// Args:
// @id name or pid
// @procArg argument to process
// @choose require a user to choose a single proc from a list
// @confirm require a user to confirm their choice
// @attach attach or detach operation
// @exactMatch require an exact match (of the name only) when id is a name
func HandleInputArg(id, procArg, rootdir string, choose, confirm, attach, exactMatch bool) (Processes, error) {
	procs := make(Processes, 0)
	var err error
	adminStatus := true
	if err := UserVerifyRootPerm(); err != nil {
		if errors.Is(err, ErrMissingAdmPriv) {
			adminStatus = false
		} else {
			return procs, err
		}
	}

	if IsNumeric(id) {
		// If the id provided is an integer, interpret it as a pid and use that pid only

		pid, err := strconv.Atoi(id)
		if err != nil {
			return procs, errPidInvalid
		}

		procs = append(procs, Process{Pid: pid})

	} else {
		// If the id provided is a name, find one or more matching procs
		// Note: An empty string is supported to pick up all procs

		if !adminStatus {
			fmt.Println("INFO: Run as root (or via sudo) to find all matching processes")
		}

		if attach {
			// Get a list of all process that match the name, regardless of status
			if procs, err = ProcessesByNameToAttach(rootdir, id, procArg, exactMatch); err != nil {
				return nil, err
			}
		} else {
			// Get a list of all processes that match the name and scope is actively attached
			if procs, err = ProcessesByNameToDetach(rootdir, id, procArg, exactMatch); err != nil {
				return nil, err
			}
		}

		if len(procs) > 1 {
			if confirm || choose {
				PrintObj([]ObjField{
					{Name: "ID", Field: "id"},
					{Name: "Pid", Field: "pid"},
					{Name: "User", Field: "user"},
					{Name: "Scoped", Field: "scoped"},
					{Name: "Command", Field: "command"},
				}, procs)
			}
			if choose {
				fmt.Println("Select an ID from the list:")
				var selection string
				fmt.Scanln(&selection)
				i, err := strconv.ParseUint(selection, 10, 32)
				i--
				if err != nil || i >= uint64(len(procs)) {
					return nil, errInvalidSelection
				}
				return Processes{procs[i]}, nil
			}
		}
	}

	return procs, nil
}

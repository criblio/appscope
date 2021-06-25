package run

import (
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"os/user"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"

	"github.com/criblio/scope/util"
	ps "github.com/mitchellh/go-ps"
	"github.com/rs/zerolog/log"
	"github.com/syndtr/gocapability/capability"
)

// Attach scopes an existing PID
func (rc *Config) Attach(args []string) {
	// Validate user has root permissions
	user, err := user.Current()
	if err != nil {
		util.ErrAndExit("Unable to get current user: %v", err)
	}
	if user.Uid != "0" {
		util.ErrAndExit("You must have administrator privileges to attach to a process")
	}
	// Validate PTRACE capability
	c, err := capability.NewPid2(0)
	if err != nil {
		util.ErrAndExit("Unable to get linux capabilities for current process: %v", err)
	}
	err = c.Load()
	if err != nil {
		util.ErrAndExit("Unable to load linux capabilities: %v", err)
	}
	if !c.Get(capability.EFFECTIVE, capability.CAP_SYS_PTRACE) {
		util.ErrAndExit("You must have PTRACE capabilities to attach to a process")
	}
	// Get PID by name if non-numeric otherwise use args[0]
	var pid int
	if !util.IsNumeric(args[0]) {
		pid = PidByName(args[0])
		args[0] = fmt.Sprint(pid)
	} else {
		pid, err = strconv.Atoi(args[0])
		if err != nil {
			util.ErrAndExit("Invalid PID: ", err)
		}
	}
	// Validate PID
	// Check PID exists and is not already being scoped
	if !PidExists(pid) {
		util.ErrAndExit("PID does not exist: \"%v\"", pid)
	}
	if PidIsScoped(pid) {
		util.ErrAndExit("Attach failed. This process is already being scoped")
	}

	// Create ldscope
	if err := createLdscope(); err != nil {
		util.ErrAndExit("error creating ldscope: %v", err)
	}
	// Normal operational, not passthrough, create directory for this run
	// Directory contains scope.yml which is configured to output to that
	// directory and has a command directory configured in that directory.
	env := os.Environ()
	if !rc.Passthrough {
		rc.setupWorkDir(args, true)
		env = append(env, "SCOPE_CONF_PATH="+filepath.Join(rc.WorkDir, "scope.yml"))
		log.Info().Bool("passthrough", rc.Passthrough).Strs("args", args).Msg("calling syscall.Exec")
	}
	if len(rc.LibraryPath) > 0 {
		// Validate path exists
		if !util.CheckDirExists(rc.LibraryPath) {
			util.ErrAndExit("Library Path does not exist: \"%s\"", rc.LibraryPath)
		}
		// Prepend "-f" [PATH] to args
		args = append([]string{"-f", rc.LibraryPath}, args...)
	}
	// Prepend "--attach" to args
	args = append([]string{"--attach"}, args...)
	if !rc.Subprocess {
		syscall.Exec(ldscopePath(), append([]string{"ldscope"}, args...), env)
	}
	cmd := exec.Command(ldscopePath(), args...)
	cmd.Env = env
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Run()
}

// PidByName gets a PID by its name
func PidByName(name string) (pid int) {
	procs, err := ps.Processes()
	if err != nil {
		util.ErrAndExit("Unable to get list of processes: %v", err)
	}
	type process struct {
		ID         int    `json:"id"`
		Uid        int    `json:"uid"`
		Pid        int    `json:"pid"`
		PPid       int    `json:"ppid"`
		Cmd        string `json:"cmd"`
		Executable string `json:"executable"`
		Scoped     bool   `json:"scoped"`
	}
	processes := make([]process, 0)
	found := 0
	lastPid := 0
	for _, p := range procs {
		if p.Executable() == name {
			found++
			lastPid = p.Pid()
			processes = append(processes, process{
				ID:     found,
				Pid:    p.Pid(),
				PPid:   p.PPid(),
				Cmd:    PidCmd(p.Pid()),
				Scoped: PidIsScoped(p.Pid()),
			})
		}
	}
	if found == 0 {
		util.ErrAndExit("Unable to find process by name")
	} else if found == 1 {
		pid = lastPid
	} else {
		fmt.Println("Found multiple processes matching that name:")
		util.PrintObj([]util.ObjField{
			{Name: "ID", Field: "id"},
			{Name: "UID", Field: "uid"},
			{Name: "PID", Field: "pid"},
			{Name: "PPID", Field: "ppid"},
			{Name: "CMD", Field: "cmd"},
			{Name: "Scoped", Field: "scoped"},
		}, processes)
		fmt.Println("Select an ID from the list:")
		var selection string
		fmt.Scanln(&selection)
		id, err := strconv.Atoi(selection)
		if err != nil || id < 1 {
			util.ErrAndExit("Invalid Selection")
		}
		pid = processes[id-1].Pid
	}
	return pid
}

// PidExists checks if a PID is valid
func PidExists(pid int) bool {
	pidPath := fmt.Sprintf("/proc/%v", pid)
	if util.CheckDirExists(pidPath) {
		return true
	}
	return false
}

// PidCmd gets the command used to start the PID
func PidCmd(pid int) string {
	pidPath := fmt.Sprintf("/proc/%v", pid)
	pidCmdFile, err := ioutil.ReadFile(pidPath + "/cmdline")
	if err != nil {
		util.ErrAndExit("cmdline does not exist for PID: \"%s\"", pid)
	}
	return string(pidCmdFile)
}

// PidIsScoped checks if a PID is being scoped
func PidIsScoped(pid int) bool {
	pidPath := fmt.Sprintf("/proc/%v", pid)
	pidMapFile, err := ioutil.ReadFile(pidPath + "/maps")
	if err != nil {
		util.ErrAndExit("Map does not exist for PID: \"%s\"", pid)
	}
	pidMap := string(pidMapFile)
	if strings.Contains(pidMap, "libscope") {
		return true
	}
	return false
}

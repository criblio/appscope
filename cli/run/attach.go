package run

import (
	"fmt"
	"os"
	"os/exec"
	"os/user"
	"path/filepath"
	"strconv"
	"syscall"

	"github.com/criblio/scope/util"
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
	// Get PID by name if non-numeric, otherwise validate/use args[0]
	var pid int
	if !util.IsNumeric(args[0]) {
		procs := util.ProcessesByName(args[0])
		if len(procs) == 1 {
			pid = procs[0].Pid
		} else if len(procs) > 1 {
			fmt.Println("Found multiple processes matching that name...")
			pid = choosePid(procs)
		} else {
			util.ErrAndExit("No process found matching that name")
		}
		args[0] = fmt.Sprint(pid)
	} else {
		pid, err = strconv.Atoi(args[0])
		if err != nil {
			util.ErrAndExit("Invalid PID: %s", err)
		}
	}
	// Check PID exists
	if !util.PidExists(pid) {
		util.ErrAndExit("PID does not exist: \"%v\"", pid)
	}
	// Check PID is not already being scoped
	if util.PidScoped(pid) {
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

// choosePid presents a user interface for selecting a PID
func choosePid(procs util.Processes) int {
	util.PrintObj([]util.ObjField{
		{Name: "ID", Field: "id"},
		{Name: "Pid", Field: "pid"},
		{Name: "User", Field: "user"},
		{Name: "Command", Field: "command"},
		{Name: "Scoped", Field: "scoped"},
	}, procs)
	fmt.Println("Select an ID from the list:")
	var selection string
	fmt.Scanln(&selection)
	i, err := strconv.Atoi(selection)
	i--
	if err != nil || i < 0 || i > len(procs) {
		util.ErrAndExit("Invalid Selection")
	}
	return procs[i].Pid
}

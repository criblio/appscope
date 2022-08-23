package run

import (
	"errors"
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

var (
	ErrGetCurrentUser           = errors.New("unable to get current user")
	ErrMissingAdmPriv           = errors.New("you must have administrator privileges to attach to a process")
	ErrGetLinuxCap              = errors.New("unable to get linux capabilities for current process")
	ErrLoadLinuxCap             = errors.New("unable to load linux capabilities for current process")
	ErrMissingPtrace            = errors.New("missing PTRACE capabilities to attach to a process")
	ErrMissingProc              = errors.New("no process found matching that name")
	ErrPidInvalid               = errors.New("invalid PID")
	ErrPidMissing               = errors.New("PID does not exist")
	ErrCreateLdscope            = errors.New("error creating ldscope")
	ErrAttachFailedAlreadyScope = errors.New("attach failed. This process is already being scoped")
	ErrLibraryNotExit           = errors.New("library Path does not exist")
	ErrInvalidSelection         = errors.New("invalid Selection")
)

// Attach scopes an existing PID
func (rc *Config) Attach(args []string) error {
	// Validate user has root permissions
	user, err := user.Current()
	if err != nil {
		return ErrGetCurrentUser
	}
	if user.Uid != "0" {
		return ErrMissingAdmPriv
	}
	// Validate PTRACE capability
	c, err := capability.NewPid2(0)
	if err != nil {
		return ErrGetLinuxCap
	}
	err = c.Load()
	if err != nil {
		return ErrLoadLinuxCap
	}
	if !c.Get(capability.EFFECTIVE, capability.CAP_SYS_PTRACE) {
		return ErrMissingPtrace
	}
	// Get PID by name if non-numeric, otherwise validate/use args[0]
	var pid int
	if !util.IsNumeric(args[0]) {
		procs, err := util.ProcessesByName(args[0])
		if err != nil {
			return err
		}
		if len(procs) == 1 {
			pid = procs[0].Pid
		} else if len(procs) > 1 {
			fmt.Println("Found multiple processes matching that name...")
			pid, err = choosePid(procs)
			if err != nil {
				return err
			}
		} else {
			return ErrMissingProc
		}
		args[0] = fmt.Sprint(pid)
	} else {
		pid, err = strconv.Atoi(args[0])
		if err != nil {
			return ErrPidInvalid
		}
	}
	// Check PID exists
	if !util.PidExists(pid) {
		return ErrPidMissing
	}
	// Check PID is not already being scoped
	if util.PidScoped(pid) {
		return ErrAttachFailedAlreadyScope
	}
	// Create ldscope
	if err := createLdscope(); err != nil {
		return ErrCreateLdscope
	}
	// Normal operational, not passthrough, create directory for this run
	// Directory contains scope.yml which is configured to output to that
	// directory and has a command directory configured in that directory.
	env := os.Environ()
	if rc.NoBreaker {
		env = append(env, "SCOPE_CRIBL_NO_BREAKER=true")
	}
	if !rc.Passthrough {
		rc.setupWorkDir(args, true)
		env = append(env, "SCOPE_CONF_PATH="+filepath.Join(rc.WorkDir, "scope.yml"))
		log.Info().Bool("passthrough", rc.Passthrough).Strs("args", args).Msg("calling syscall.Exec")
	}
	if len(rc.LibraryPath) > 0 {
		// Validate path exists
		if !util.CheckDirExists(rc.LibraryPath) {
			return ErrLibraryNotExit
		}
		// Prepend "-f" [PATH] to args
		args = append([]string{"-f", rc.LibraryPath}, args...)
	}
	// Prepend "--attach" to args
	args = append([]string{"--attach"}, args...)
	if !rc.Subprocess {
		return syscall.Exec(ldscopePath(), append([]string{"ldscope"}, args...), env)
	}
	cmd := exec.Command(ldscopePath(), args...)
	cmd.Env = env
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

// choosePid presents a user interface for selecting a PID
func choosePid(procs util.Processes) (int, error) {
	util.PrintObj([]util.ObjField{
		{Name: "ID", Field: "id"},
		{Name: "Pid", Field: "pid"},
		{Name: "User", Field: "user"},
		{Name: "Scoped", Field: "scoped"},
		{Name: "Command", Field: "command"},
	}, procs)
	fmt.Println("Select an ID from the list:")
	var selection string
	fmt.Scanln(&selection)
	i, err := strconv.ParseUint(selection, 10, 32)
	i--
	if err != nil || i >= uint64(len(procs)) {
		return -1, ErrInvalidSelection
	}
	return procs[i].Pid, nil
}

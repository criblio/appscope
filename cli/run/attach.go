package run

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"

	"github.com/criblio/scope/loader"
	"github.com/criblio/scope/util"
	"github.com/rs/zerolog/log"
	"github.com/syndtr/gocapability/capability"
)

var (
	errGetLinuxCap      = errors.New("unable to get linux capabilities for current process")
	errLoadLinuxCap     = errors.New("unable to load linux capabilities for current process")
	errMissingPtrace    = errors.New("missing PTRACE capabilities to attach to a process")
	errMissingProc      = errors.New("no process found matching that name")
	errPidInvalid       = errors.New("invalid PID")
	errPidMissing       = errors.New("PID does not exist")
	errCreateLdscope    = errors.New("error creating ldscope")
	errAlreadyScope     = errors.New("attach failed. This process is already being scoped")
	errLibraryNotExist  = errors.New("library Path does not exist")
	errInvalidSelection = errors.New("invalid Selection")
)

// Attach scopes an existing PID
func (rc *Config) Attach(args []string) error {
	// Validate user has root permissions
	if err := util.UserVerifyRootPerm(); err != nil {
		return err
	}
	// Validate PTRACE capability
	c, err := capability.NewPid2(0)
	if err != nil {
		return errGetLinuxCap
	}
	err = c.Load()
	if err != nil {
		return errLoadLinuxCap
	}
	if !c.Get(capability.EFFECTIVE, capability.CAP_SYS_PTRACE) {
		return errMissingPtrace
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
			return errMissingProc
		}
		args[0] = fmt.Sprint(pid)
	} else {
		pid, err = strconv.Atoi(args[0])
		if err != nil {
			return errPidInvalid
		}
	}
	// Check PID exists
	if !util.PidExists(pid) {
		return errPidMissing
	}
	// Check PID is not already being scoped
	if util.PidScoped(pid) {
		return errAlreadyScope
	}
	// Create ldscope
	if err := CreateLdscope(); err != nil {
		return errCreateLdscope
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
			return errLibraryNotExist
		}
		// Prepend "-f" [PATH] to args
		args = append([]string{"-f", rc.LibraryPath}, args...)
	}
	sL := loader.ScopeLoader{Path: LdscopePath()}
	if !rc.Subprocess {
		return sL.Attach(args, env)
	}
	_, err = sL.AttachSubProc(args, env)
	return err
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
		return -1, errInvalidSelection
	}
	return procs[i].Pid, nil
}

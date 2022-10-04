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
	errNotScoped        = errors.New("detach failed. This process is not being scoped")
	errLibraryNotExist  = errors.New("library Path does not exist")
	errInvalidSelection = errors.New("invalid Selection")
)

// Attach scopes an existing PID
func (rc *Config) Attach(args []string) error {
	pid, err := handleInputArg(args[0])
	if err != nil {
		return err
	}
	args[0] = fmt.Sprint(pid)
	var reattach bool
	// Check PID is not already being scoped
	if !util.PidScoped(pid) {
		// Validate user has root permissions
		if err := util.UserVerifyRootPerm(); err != nil {
			return err
		}
		// Validate PTRACE capability
		c, err := capability.NewPid2(0)
		if err != nil {
			return errGetLinuxCap
		}

		if err = c.Load(); err != nil {
			return errLoadLinuxCap
		}

		if !c.Get(capability.EFFECTIVE, capability.CAP_SYS_PTRACE) {
			return errMissingPtrace
		}
	} else {
		// Reattach because process contains our library
		reattach = true
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
	if reattach {
		env = append(env, "SCOPE_CONF_RELOAD="+filepath.Join(rc.WorkDir, "scope.yml"))
	}

	ld := loader.ScopeLoader{Path: LdscopePath()}
	if !rc.Subprocess {
		return ld.Attach(args, env)
	}
	_, err = ld.AttachSubProc(args, env)
	return err
}

// Detach unscopes an existing PID
func (rc *Config) Detach(args []string) error {
	pid, err := handleInputArg(args[0])
	if err != nil {
		return err
	}
	args[0] = fmt.Sprint(pid)

	// Check PID is already being scoped
	if !util.PidScoped(pid) {
		return errNotScoped
	}

	env := os.Environ()

	// Create ldscope
	if err := CreateLdscope(); err != nil {
		return errCreateLdscope
	}

	ld := loader.ScopeLoader{Path: LdscopePath()}
	if !rc.Subprocess {
		return ld.Detach(args, env)
	}
	_, err = ld.DetachSubProc(args, env)
	return err
}

// handleInputArg handles the input argument (process id/name)
func handleInputArg(InputArg string) (int, error) {
	// Get PID by name if non-numeric, otherwise validate/use InputArg
	var pid int
	var err error
	if !util.IsNumeric(InputArg) {
		procs, err := util.ProcessesByName(InputArg)
		if err != nil {
			return -1, err
		}
		if len(procs) == 1 {
			pid = procs[0].Pid
		} else if len(procs) > 1 {
			// user interface for selecting a PID
			fmt.Println("Found multiple processes matching that name...")
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
			pid = procs[i].Pid
		} else {
			return -1, errMissingProc
		}
	} else {
		pid, err = strconv.Atoi(InputArg)
		if err != nil {
			return -1, errPidInvalid
		}
	}

	// Check PID exists
	if !util.PidExists(pid) {
		return -1, errPidMissing
	}

	return pid, nil
}

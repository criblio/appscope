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
	errGetLinuxCap       = errors.New("unable to get linux capabilities for current process")
	errLoadLinuxCap      = errors.New("unable to load linux capabilities for current process")
	errMissingPtrace     = errors.New("missing PTRACE capabilities to attach to a process")
	errMissingScopedProc = errors.New("no scoped process found matching that name")
	errMissingProc       = errors.New("no process found matching that name")
	errPidInvalid        = errors.New("invalid PID")
	errPidMissing        = errors.New("PID does not exist")
	errNotScoped         = errors.New("detach failed. This process is not being scoped")
	errLibraryNotExist   = errors.New("library Path does not exist")
	errInvalidSelection  = errors.New("invalid Selection")
	errNoScopedProcs     = errors.New("no scoped processes found")
	errDetachingMultiple = errors.New("at least one error found when detaching from all. See logs")
)

// Attach scopes an existing PID
func (rc *Config) Attach(args []string) error {
	pid, err := HandleInputArg(args[0], true, false, true)
	if err != nil {
		return err
	}
	args[0] = fmt.Sprint(pid)
	var reattach bool
	// Check PID is not already being scoped
	status, err := util.PidScopeStatus(pid)
	if err != nil {
		return err
	}

	if status == util.Disable || status == util.Setup {
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

	env := os.Environ()

	// Disable detection of a scope filter file with this command
	env = append(env, "SCOPE_FILTER=false")

	// Disable cribl event breaker with this command
	if rc.NoBreaker {
		env = append(env, "SCOPE_CRIBL_NO_BREAKER=true")
	}

	// Normal operational, create a directory for this run.
	// Directory contains scope.yml which is configured to output to that
	// directory and has a command directory configured in that directory.
	rc.setupWorkDir(args, true)
	env = append(env, "SCOPE_CONF_PATH="+filepath.Join(rc.WorkDir, "scope.yml"))

	// Check the attached process mnt namespace.
	// If it is different from the CLI mnt namespace:
	// - create working directory in the attached process mnt namespace
	// - replace the working directory in the CLI mnt namespace with symbolic
	//   link to working directory created in previous step
	refNsPid := util.PidGetRefPidForMntNamespace(pid)
	if refNsPid != -1 {
		env = append(env, "SCOPE_HOST_WORKDIR_PATH="+rc.WorkDir)
	}

	// Handle custom library path
	if len(rc.LibraryPath) > 0 {
		if !util.CheckDirExists(rc.LibraryPath) {
			return errLibraryNotExist
		}
		args = append([]string{"-f", rc.LibraryPath}, args...)
	}

	if reattach {
		env = append(env, "SCOPE_CONF_RELOAD="+filepath.Join(rc.WorkDir, "scope.yml"))
	}

	ld := loader.New()
	if !rc.Subprocess {
		err = ld.Attach(args, env)
	} else {
		_, err = ld.AttachSubProc(args, env)
	}

	// Replace the working directory with symbolic link in case of successful attach
	if err == nil {
		if refNsPid != -1 {
			os.RemoveAll(rc.WorkDir)
			os.Symlink(filepath.Join("/proc", fmt.Sprint(refNsPid), "root", rc.WorkDir), rc.WorkDir)
		}
	}

	return err
}

// DetachAll provides the option to detach from all Scoped processes
func (rc *Config) DetachAll(args []string, prompt bool) error {
	adminStatus := true
	if err := util.UserVerifyRootPerm(); err != nil {
		if errors.Is(err, util.ErrMissingAdmPriv) {
			adminStatus = false
		} else {
			return err
		}
	}
	if !adminStatus {
		fmt.Println("INFO: Run as root (or via sudo) to see all matching processes")
	}

	procs, err := util.ProcessesToDetach()
	if err != nil {
		return err
	}

	if len(procs) > 0 {
		if prompt {
			// user interface for selecting a PID
			util.PrintObj([]util.ObjField{
				{Name: "ID", Field: "id"},
				{Name: "Pid", Field: "pid"},
				{Name: "User", Field: "user"},
				{Name: "Command", Field: "command"},
			}, procs)

			if !util.Confirm("Are your sure you want to detach from all of these processes?") {
				fmt.Println("info: canceled")
				return nil
			}
		}
	} else {
		return errNoScopedProcs
	}

	errorsDetachingMultiple := false
	for _, proc := range procs {
		tmpArgs := append([]string{fmt.Sprint(proc.Pid)}, args...)
		if err := rc.detach(tmpArgs, proc.Pid); err != nil {
			log.Error().Err(err)
			errorsDetachingMultiple = true
		}
	}
	if errorsDetachingMultiple {
		return errDetachingMultiple
	}

	return nil
}

// DetachSingle unscopes an existing PID
func (rc *Config) DetachSingle(args []string) error {
	pid, err := HandleInputArg(args[0], false, false, true)
	if err != nil {
		return err
	}
	args[0] = fmt.Sprint(pid)

	// Check PID is already being scoped
	status, err := util.PidScopeStatus(pid)
	if err != nil {
		return err
	} else if status != util.Active {
		return errNotScoped
	}

	return rc.detach(args, pid)
}

func (rc *Config) detach(args []string, pid int) error {
	env := os.Environ()
	ld := loader.New()
	if !rc.Subprocess {
		return ld.Detach(args, env)
	}
	out, err := ld.DetachSubProc(args, env)
	fmt.Print(out)

	return err
}

// HandleInputArg handles the input argument (process id/name)
func HandleInputArg(InputArg string, toAttach, singleProcMenu, warn bool) (int, error) {
	// Get PID by name if non-numeric, otherwise validate/use InputArg
	var pid int
	var err error
	var procs util.Processes
	if util.IsNumeric(InputArg) {
		pid, err = strconv.Atoi(InputArg)
		if err != nil {
			return -1, errPidInvalid
		}
	} else {
		adminStatus := true
		if err := util.UserVerifyRootPerm(); err != nil {
			if errors.Is(err, util.ErrMissingAdmPriv) {
				adminStatus = false
			} else {
				return -1, err
			}
		}

		if toAttach {
			procs, err = util.ProcessesByNameToAttach(InputArg)
		} else {
			procs, err = util.ProcessesByNameToDetach(InputArg)
		}

		if err != nil {
			return -1, err
		}
		if len(procs) == 1 && !singleProcMenu {
			pid = procs[0].Pid
		} else if len(procs) >= 1 {
			if !adminStatus && warn {
				fmt.Println("INFO: Run as root (or via sudo) to see all matching processes")
			}

			// user interface for selecting a PID
			fmt.Println("Select a process...")
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
			if !adminStatus && warn {
				fmt.Println("INFO: Run as root (or via sudo) to see all matching processes")
			}
			if toAttach {
				return -1, errMissingProc
			}
			return -1, errMissingScopedProc
		}
	}

	// Check PID exists
	if !util.PidExists(pid) {
		return -1, errPidMissing
	}

	return pid, nil
}

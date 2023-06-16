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
func (rc *Config) Attach(args []string) (int, error) {
	pid, err := HandleInputArg(rc.Rootdir, args[0], true, false, true)
	if err != nil {
		return pid, err
	}
	args[0] = fmt.Sprint(pid)
	var reattach bool
	// Check PID is not already being scoped
	status, err := util.PidScopeStatus(rc.Rootdir, pid)
	if err != nil {
		return pid, err
	}

	if status == util.Disable || status == util.Setup {
		// Validate user has root permissions
		if err := util.UserVerifyRootPerm(); err != nil {
			return pid, err
		}
		// Validate PTRACE capability
		c, err := capability.NewPid2(0)
		if err != nil {
			return pid, errGetLinuxCap
		}

		if err = c.Load(); err != nil {
			return pid, errLoadLinuxCap
		}

		if !c.Get(capability.EFFECTIVE, capability.CAP_SYS_PTRACE) {
			return pid, errMissingPtrace
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
	refNsPid := util.PidGetRefPidForMntNamespace(rc.Rootdir, pid)
	if refNsPid != -1 {
		env = append(env, "SCOPE_HOST_WORKDIR_PATH="+rc.WorkDir)
	}

	// Handle custom library path
	if len(rc.LibraryPath) > 0 {
		if !util.CheckDirExists(rc.LibraryPath) {
			return pid, errLibraryNotExist
		}
		args = append([]string{"-f", rc.LibraryPath}, args...)
	}

	if reattach {
		env = append(env, "SCOPE_CONF_RELOAD="+filepath.Join(rc.WorkDir, "scope.yml"))
	}

	if rc.Rootdir != "" {
		args = append(args, []string{"--rootdir", rc.Rootdir}...)
	}

	ld := loader.New()
	stdoutStderr, err := ld.AttachSubProc(args, env)
	util.Warn(stdoutStderr)
	if err != nil {
		return pid, err
	}

	// Replace the working directory files with symbolic links in case of successful attach
	// where the target ns is different to the origin ns

	eventsFilePath := filepath.Join(rc.WorkDir, "events.json")
	metricsFilePath := filepath.Join(rc.WorkDir, "metrics.json")
	logsFilePath := filepath.Join(rc.WorkDir, "libscope.log")
	payloadsDirPath := filepath.Join(rc.WorkDir, "payloads")

	if rc.Rootdir != "" {
		os.Remove(eventsFilePath)
		os.Remove(metricsFilePath)
		os.Remove(logsFilePath)
		os.RemoveAll(payloadsDirPath)
		os.Symlink(filepath.Join(rc.Rootdir, "/proc", fmt.Sprint(pid), "root", eventsFilePath), eventsFilePath)
		os.Symlink(filepath.Join(rc.Rootdir, "/proc", fmt.Sprint(pid), "root", metricsFilePath), metricsFilePath)
		os.Symlink(filepath.Join(rc.Rootdir, "/proc", fmt.Sprint(pid), "root", logsFilePath), logsFilePath)
		os.Symlink(filepath.Join(rc.Rootdir, "/proc", fmt.Sprint(pid), "root", payloadsDirPath), payloadsDirPath)

	} else if refNsPid != -1 {
		// Child namespace
		os.Remove(eventsFilePath)
		os.Remove(metricsFilePath)
		os.Remove(logsFilePath)
		os.RemoveAll(payloadsDirPath)
		os.Symlink(filepath.Join("/proc", fmt.Sprint(refNsPid), "root", eventsFilePath), eventsFilePath)
		os.Symlink(filepath.Join("/proc", fmt.Sprint(refNsPid), "root", metricsFilePath), metricsFilePath)
		os.Symlink(filepath.Join("/proc", fmt.Sprint(refNsPid), "root", logsFilePath), logsFilePath)
		os.Symlink(filepath.Join("/proc", fmt.Sprint(refNsPid), "root", payloadsDirPath), payloadsDirPath)
	}

	return pid, nil
}

// DetachAll provides the option to detach from all Scoped processes
func (rc *Config) DetachAll(prompt bool) error {
	adminStatus := true
	if err := util.UserVerifyRootPerm(); err != nil {
		if errors.Is(err, util.ErrMissingAdmPriv) {
			adminStatus = false
		} else {
			return err
		}
	}
	if !adminStatus {
		util.Warn("INFO: Run as root (or via sudo) to see all matching processes")
	}

	procs, err := util.ProcessesToDetach(rc.Rootdir)
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
		if err := rc.detach(proc.Pid); err != nil {
			log.Error().Err(err)
			errorsDetachingMultiple = true
		}
	}
	if errorsDetachingMultiple {
		return errDetachingMultiple
	}

	return nil
}

// DetachSingle unscopes an existing scoped process, identified by name or pid
func (rc *Config) DetachSingle(id string) error {
	pid, err := HandleInputArg(rc.Rootdir, id, false, true, true)
	if err != nil {
		return err
	}

	// Check PID is already being scoped
	status, err := util.PidScopeStatus(rc.Rootdir, pid)
	if err != nil {
		return err
	} else if status != util.Active {
		return errNotScoped
	}

	return rc.detach(pid)
}

func (rc *Config) detach(pid int) error {
	args := make([]string, 0)
	args = append(args, fmt.Sprint(pid))

	if rc.Rootdir != "" {
		args = append(args, []string{"--rootdir", rc.Rootdir}...)
	}

	env := os.Environ()
	ld := loader.New()
	stdoutStderr, err := ld.DetachSubProc(args, env)
	util.Warn(stdoutStderr)
	if err != nil {
		return err
	}

	return nil
}

// HandleInputArg handles the input argument (process id/name)
func HandleInputArg(rootdir, InputArg string, toAttach, singleProcMenu, warn bool) (int, error) {
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
			procs, err = util.ProcessesByNameToAttach(rootdir, InputArg)
		} else {
			procs, err = util.ProcessesByNameToDetach(rootdir, InputArg)
		}

		if err != nil {
			return -1, err
		}
		if len(procs) == 1 && !singleProcMenu {
			pid = procs[0].Pid
		} else if len(procs) >= 1 {
			if !adminStatus && warn {
				util.Warn("INFO: Run as root (or via sudo) to see all matching processes")
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
				util.Warn("INFO: Run as root (or via sudo) to see all matching processes")
			}
			if toAttach {
				return -1, errMissingProc
			}
			return -1, errMissingScopedProc
		}
	}

	// Check PID exists
	if !util.PidExists(rootdir, pid) {
		return -1, errPidMissing
	}

	return pid, nil
}

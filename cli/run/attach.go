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
	errAttachingMultiple = errors.New("at least one error found when attaching to more than 1 process. See logs")
	errDetachingMultiple = errors.New("at least one error found when detaching from more than 1 process. See logs")
)

// AttachDetachMultiple allows a user to attach to or detach from one or more processes by name or pid
// If the id provided is a pid, that single pid will be attached/detached
// If the id provided is a name, all processes matching that name will be attached/detached
// unless the @choose argument is true, which will allow the user to choose a single process
// NOTE: The responsibility of this function is not to check if its possible to attach/detach
// It's only to prepare a list of procs and let attach/detach handle the rest
func (rc *Config) AttachDetachMultiple(id string, choose, confirm, attach bool) (util.Processes, error) {
	// Differentiate between attach and detach actions
	function := rc.attach
	actionString := "attach to"
	noProcsErr := errMissingProc
	multipleErr := errAttachingMultiple
	if !attach {
		function = rc.detach
		actionString = "detach from"
		noProcsErr = errNoScopedProcs
		multipleErr = errDetachingMultiple
	}

	procs := make(util.Processes, 0)
	var err error
	adminStatus := true
	if err := util.UserVerifyRootPerm(); err != nil {
		if errors.Is(err, util.ErrMissingAdmPriv) {
			adminStatus = false
		} else {
			return procs, err
		}
	}

	if util.IsNumeric(id) {
		// If the id provided is an integer, interpret it as a pid and use that pid only

		pid, err := strconv.Atoi(id)
		if err != nil {
			return procs, errPidInvalid
		}

		procs = append(procs, util.Process{Pid: pid})

	} else {
		// If the id provided is a name, find one or more matching procs

		if !adminStatus {
			fmt.Println("INFO: Run as root (or via sudo) to find all matching processes")
		}

		procs, err = HandleInputArg(rc.Rootdir, id, false, choose)
		if err != nil {
			return procs, err
		}
	}

	if len(procs) == 0 {
		return procs, noProcsErr
	}
	if len(procs) == 1 {
		return procs, function(procs[0].Pid)
	}

	// len(procs) is > 1
	if confirm && !util.Confirm(fmt.Sprintf("Are your sure you want to %s all of these processes?", actionString)) {
		fmt.Println("info: canceled")
		return procs, nil
	}

	errors := false
	for _, proc := range procs {
		if err = function(proc.Pid); err != nil {
			log.Error().Err(err)
			errors = true
		}
	}
	if errors {
		return procs, multipleErr
	}

	return procs, nil
}

// NOTE: The responsibility of this function is to check if its possible to attach
// and then perform the attach if so
func (rc *Config) attach(pid int) error {
	env := os.Environ()
	ld := loader.New()

	var reattach bool

	status, err := util.PidScopeStatus(rc.Rootdir, pid)
	if err != nil {
		return err
	}
	if status == util.Disable || status == util.Setup {
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

	args := []string{fmt.Sprint(pid)}

	if rc.Rootdir != "" {
		args = append(args, []string{"--rootdir", rc.Rootdir}...)
	}

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
			return errLibraryNotExist
		}
		args = append([]string{"-f", rc.LibraryPath}, args...)
	}

	if reattach {
		env = append(env, "SCOPE_CONF_RELOAD="+filepath.Join(rc.WorkDir, "scope.yml"))
	}

	if rc.Subprocess {
		out, err := ld.AttachSubProc(args, env)
		if err != nil {
			return err
		}
		fmt.Print(out)
	} else {
		if err = ld.Attach(args, env); err != nil {
			return err
		}
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

	return nil
}

// NOTE: The responsibility of this function is to check if its possible to detach
// and then perform the detach if so
func (rc *Config) detach(pid int) error {
	env := os.Environ()
	ld := loader.New()

	status, err := util.PidScopeStatus(rc.Rootdir, pid)
	if err != nil {
		return err
	}
	if status != util.Active {
		return errNotScoped
	}

	args := []string{fmt.Sprint(pid)}

	if rc.Rootdir != "" {
		args = append(args, []string{"--rootdir", rc.Rootdir}...)
	}

	if rc.Subprocess {
		out, err := ld.DetachSubProc(args, env)
		fmt.Print(out)
		return err
	}
	return ld.Detach(args, env)
}

// HandleInputArg handles the input argument (process name) and returns an array of processes
func HandleInputArg(rootdir, inputArg string, toAttach, choose bool) (util.Processes, error) {
	var err error
	var procs util.Processes

	if toAttach {
		// Get a list of all process that match the name, regardless of status
		if procs, err = util.ProcessesByNameToAttach(rootdir, inputArg); err != nil {
			return nil, err
		}
	} else {
		// Get a list of all processes that match the name and scope is actively attached
		if procs, err = util.ProcessesByNameToDetach(rootdir, inputArg); err != nil {
			return nil, err
		}
	}

	if len(procs) > 1 && choose {
		// User interface for selecting a PID
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
			return nil, errInvalidSelection
		}
		return util.Processes{procs[i]}, nil
	}

	return procs, nil
}

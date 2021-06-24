package run

import (
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"os/user"
	"path/filepath"
	"strings"
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
	// Validate PID exists
	pidPath := fmt.Sprintf("/proc/%s", args[0])
	if !util.CheckDirExists(pidPath) {
		util.ErrAndExit("PID does not exist: \"%s\"", args[0])
	}
	// Validate the process isn't already being scoped
	pidMapFile, err := ioutil.ReadFile(pidPath + "/maps")
	if err != nil {
		util.ErrAndExit("Map does not exist for PID: \"%s\"", args[0])
	}
	pidMap := string(pidMapFile)
	if strings.Contains(pidMap, "libscope") {
		util.ErrAndExit("Attach failed. This process is already being scoped")
	}
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

package loader

import (
	"os"
	"os/exec"
	"strconv"
	"syscall"
)

// Loader represents ldscope object
type ScopeLoader struct {
	Path string
}

func (sL *ScopeLoader) ConfigureHost(filterFilePath string) (string, error) {
	return sL.RunSubProc([]string{"--configure", filterFilePath}, os.Environ())
}

func (sL *ScopeLoader) ConfigureContainer(filterFilePath string, cpid int) (string, error) {
	return sL.RunSubProc([]string{"--configure", filterFilePath, "--namespace", strconv.Itoa(cpid)}, os.Environ())
}

func (sL *ScopeLoader) ServiceHost(serviceName string) (string, error) {
	return sL.RunSubProc([]string{"--service", serviceName}, os.Environ())
}

func (sL *ScopeLoader) ServiceContainer(serviceName string, cpid int) (string, error) {
	return sL.RunSubProc([]string{"--service", serviceName, "--namespace", strconv.Itoa(cpid)}, os.Environ())
}

func (sL *ScopeLoader) Patch(libraryPath string) (string, error) {
	return sL.RunSubProc([]string{"--patch", libraryPath}, os.Environ())
}

func (sL *ScopeLoader) Run(args []string, env []string) error {
	args = append([]string{"ldscope"}, args...)
	return syscall.Exec(sL.Path, args, env)
}

func (sL *ScopeLoader) RunSubProc(args []string, env []string) (string, error) {
	cmd := exec.Command(sL.Path, args...)
	cmd.Env = env
	stdoutStderr, err := cmd.CombinedOutput()
	return string(stdoutStderr[:]), err
}

func (sL *ScopeLoader) Attach(args []string, env []string) error {
	args = append([]string{"--attach"}, args...)
	return sL.Run(args, env)
}

func (sL *ScopeLoader) AttachSubProc(args []string, env []string) (string, error) {
	args = append([]string{"--attach"}, args...)
	return sL.RunSubProc(args, env)
}

// Detach transforms the calling process into a ldscope detach operation
func (sL *ScopeLoader) Detach(args []string, env []string) error {
	args = append([]string{"--detach"}, args...)
	return sL.Run(args, env)
}

// DetachSubProc runs a ldscope detach as a seperate process
func (sL *ScopeLoader) DetachSubProc(args []string, env []string) (string, error) {
	args = append([]string{"--detach"}, args...)
	return sL.RunSubProc(args, env)
}

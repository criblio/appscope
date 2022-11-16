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

// - Setup /etc/profile.d/scope.sh on host
// - Extract libscope.so to /usr/lib/appscope/<version>/libscope.so /tmp/appscope/<version>/libscope.so on host
// - Extract filter input to /usr/lib/appscope/scope_filter or /tmp/appscope/scope_filter on host
func (sL *ScopeLoader) ConfigureHost(filterFilePath string) (string, error) {
	return sL.RunSubProc([]string{"--configure", filterFilePath}, os.Environ())
}

// - Setup /etc/profile.d/scope.sh in containers
// - Extract libscope.so to /usr/lib/appscope/<version>/libscope.so or /tmp/appscope/<version>/libscope.so in containers
// - Extract filter input to /usr/lib/appscope/scope_filter or /tmp/appscope/scope_filter in containers
func (sL *ScopeLoader) ConfigureContainer(filterFilePath string, cpid int) (string, error) {
	return sL.RunSubProc([]string{"--configure", filterFilePath, "--namespace", strconv.Itoa(cpid)}, os.Environ())
}

// - Remove /etc/profile.d/scope.sh on host
// - Remove filter input from /usr/lib/appscope/scope_filter or /tmp/appscope/scope_filter on host
func (sL *ScopeLoader) UnconfigureHost() (string, error) {
	return sL.RunSubProc([]string{"--unconfigure"}, os.Environ())
}

// - Remove /etc/profile.d/scope.sh in containers
// - Remove filter input from /usr/lib/appscope/scope_filter or /tmp/appscope/scope_filter in containers
func (sL *ScopeLoader) UnconfigureContainer(cpid int) (string, error) {
	return sL.RunSubProc([]string{"--unconfigure", "--namespace", strconv.Itoa(cpid)}, os.Environ())
}

// - Modify the relevant service configurations to preload /usr/lib/appscope/<version>/libscope.so or /tmp/appscope/<version>/libscope.so on the host
func (sL *ScopeLoader) ServiceHost(serviceName string) (string, error) {
	return sL.RunSubProc([]string{"--service", serviceName}, os.Environ())
}

// - Modify the relevant service configurations to preload /usr/lib/appscope/<version>/libscope.so or /tmp/appscope/<version>/libscope.so in containers
func (sL *ScopeLoader) ServiceContainer(serviceName string, cpid int) (string, error) {
	return sL.RunSubProc([]string{"--service", serviceName, "--namespace", strconv.Itoa(cpid)}, os.Environ())
}

// - Modify the relevant service configurations to NOT preload /usr/lib/appscope/<version>/libscope.so or /tmp/appscope/<version>/libscope.so on the host
func (sL *ScopeLoader) UnserviceHost() (string, error) {
	return sL.RunSubProc([]string{"--unservice"}, os.Environ())
}

// - Modify the relevant service configurations to NOT preload /usr/lib/appscope/<version>/libscope.so or /tmp/appscope/<version>/libscope.so in containers
func (sL *ScopeLoader) UnserviceContainer(cpid int) (string, error) {
	return sL.RunSubProc([]string{"--unservice", "--namespace", strconv.Itoa(cpid)}, os.Environ())
}

// - Attach to a process on the host or in containers
func (sL *ScopeLoader) Attach(args []string, env []string) error {
	args = append([]string{"--attach"}, args...)
	return sL.Run(args, env)
}

// - Attach to a process on the host or in containers
func (sL *ScopeLoader) AttachSubProc(args []string, env []string) (string, error) {
	args = append([]string{"--attach"}, args...)
	return sL.RunSubProc(args, env)
}

// Used when scope has detected that we are running the `scope start` command inside a container
// Tell ldscope to run the `scope start` command on the host instead
func (sL *ScopeLoader) StartHost() (string, error) {
	return sL.RunSubProc([]string{"--starthost"}, os.Environ())
}

// Used when scope has detected that we are running the `scope stop` command inside a container
// Tell ldscope to run the `scope stop` command on the host instead
func (sL *ScopeLoader) StopHost() (string, error) {
	return sL.RunSubProc([]string{"--stophost"}, os.Environ())
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

func (sL *ScopeLoader) Patch(libraryPath string) (string, error) {
	return sL.RunSubProc([]string{"--patch", libraryPath}, os.Environ())
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

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

// If non-root:
// - Extract libscope.so to /tmp/libscope.so on host
// - Extract filter input to /tmp/scope_filter.yml on host
// If root:
// - Setup /etc/profile.d/scope.sh on host
// - Extract libscope.so to /usr/lib/appscope/libscope.so on host
// - Extract filter input to /usr/lib/appscope/scope_filter.yml on host
func (sL *ScopeLoader) ConfigureHost(filterFilePath string) (string, error) {
	return sL.RunSubProc([]string{"--configure", filterFilePath}, os.Environ())
}

// If non-root:
// - Extract libscope.so to /tmp/libscope.so in containers
// - Extract filter input to /tmp/scope_filter.yml in containers
// If root:
// - Setup /etc/profile.d/scope.sh in containers
// - Extract libscope.so to /usr/lib/appscope/libscope.so in containers
// - Extract filter input to /usr/lib/appscope/scope_filter.yml in containers
func (sL *ScopeLoader) ConfigureContainer(filterFilePath string, cpid int) (string, error) {
	return sL.RunSubProc([]string{"--configure", filterFilePath, "--namespace", strconv.Itoa(cpid)}, os.Environ())
}

// If root:
// - Modify the relevant service configurations to preload /usr/lib/appscope/libscope.so on the host
// Add service configurations on host
func (sL *ScopeLoader) ServiceHost(serviceName string) (string, error) {
	return sL.RunSubProc([]string{"--service", serviceName}, os.Environ())
}

// If root:
// - Modify the relevant service configurations to preload /usr/lib/appscope/libscope.so in containers
func (sL *ScopeLoader) ServiceContainer(serviceName string, cpid int) (string, error) {
	return sL.RunSubProc([]string{"--service", serviceName, "--namespace", strconv.Itoa(cpid)}, os.Environ())
}

// If root:
// - Attach to a process on the host or in containers
func (sL *ScopeLoader) Attach(args []string, env []string) error {
	args = append([]string{"--attach"}, args...)
	return sL.Run(args, env)
}

// If root:
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

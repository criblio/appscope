package loader

import (
	"os"
	"os/exec"
	"strconv"
	"syscall"

	"github.com/criblio/scope/util"
)

// Loader represents scope loader object
type ScopeLoader struct {
	Path string
}

// Initialize the loader
func New() ScopeLoader {
	// Get path to this scope executable
	exPath, err := os.Executable()
	if err != nil {
		util.ErrAndExit("error getting executable path, %v", err)
	}

	return ScopeLoader{
		Path: exPath,
	}
}

// Filter Command
func (sL *ScopeLoader) Filter(tmpPath, rootdir string) (string, error) {
	args := make([]string, 0)
	args = append(args, "--filter")
	args = append(args, tmpPath)
	if rootdir != "" {
		args = append(args, "--rootdir")
		args = append(args, rootdir)
	}
	return sL.RunSubProc(args, os.Environ())
}

// Preload Command
func (sL *ScopeLoader) Preload(path, rootdir string) (string, error) {
	args := make([]string, 0)
	args = append(args, "--preload")
	args = append(args, path)
	if rootdir != "" {
		args = append(args, "--rootdir")
		args = append(args, rootdir)
	}
	return sL.RunSubProc(args, os.Environ())
}

// Mount Command
func (sL *ScopeLoader) Mount(source, dest, rootdir string) (string, error) {
	args := make([]string, 0)
	args = append(args, "--mount")
	args = append(args, source+","+dest)
	if rootdir != "" {
		args = append(args, "--rootdir")
		args = append(args, rootdir)
	}
	return sL.RunSubProc(args, os.Environ())
}

// Install Command
// - Extract libscope.so to /usr/lib/appscope/<version>/libscope.so /tmp/appscope/<version>/libscope.so locally
func (sL *ScopeLoader) Install(rootdir string) (string, error) {
	args := make([]string, 0)
	args = append(args, "--install")
	if rootdir != "" {
		args = append(args, "--rootdir")
		args = append(args, rootdir)
	}
	return sL.RunSubProc(args, os.Environ())
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
	args = append([]string{"--ldattach"}, args...)
	return sL.ForkAndRun(args, env)
}

// - Attach to a process on the host or in containers
func (sL *ScopeLoader) AttachSubProc(args []string, env []string) (string, error) {
	args = append([]string{"--ldattach"}, args...)
	return sL.RunSubProc(args, env)
}

func (sL *ScopeLoader) Patch(libraryPath string) (string, error) {
	return sL.RunSubProc([]string{"--patch", libraryPath}, os.Environ())
}

// Detach transforms the calling process into a scope detach operation
func (sL *ScopeLoader) Detach(args []string, env []string) error {
	args = append([]string{"--lddetach"}, args...)
	return sL.Run(args, env)
}

// DetachSubProc runs a scope detach as a seperate process
func (sL *ScopeLoader) DetachSubProc(args []string, env []string) (string, error) {
	args = append([]string{"--lddetach"}, args...)
	return sL.RunSubProc(args, env)
}

// Passthrough scopes a process
func (sL *ScopeLoader) Passthrough(args []string, env []string) error {
	args = append([]string{"--passthrough"}, args...)
	return sL.Run(args, env)
}

// PassthroughSubProc scopes a process as a seperate process
func (sL *ScopeLoader) PassthroughSubProc(args []string, env []string) (string, error) {
	args = append([]string{"--passthrough"}, args...)
	return sL.RunSubProc(args, env)
}

func (sL *ScopeLoader) Run(args []string, env []string) error {
	args = append([]string{"scope"}, args...)
	return syscall.Exec(sL.Path, args, env)
}

func (sL *ScopeLoader) ForkAndRun(args []string, env []string) error {
	args = append([]string{"scope"}, args...)

	pid, err := syscall.ForkExec(sL.Path, args, &syscall.ProcAttr{
		Env: env,
		Files: []uintptr{uintptr(syscall.Stdin),
			uintptr(syscall.Stdout),
			uintptr(syscall.Stderr)}})

	// Child has exec'ed. Below is the Parent path
	if err != nil {
		return err
	}
	proc, suberr := os.FindProcess(pid)
	if suberr != nil {
		return err
	}
	_, suberr = proc.Wait()
	if suberr != nil {
		return err
	}

	return err
}

func (sL *ScopeLoader) RunSubProc(args []string, env []string) (string, error) {
	cmd := exec.Command(sL.Path, args...)
	cmd.Env = env
	stdoutStderr, err := cmd.CombinedOutput()
	return string(stdoutStderr[:]), err
}

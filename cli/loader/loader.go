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

func (sL *ScopeLoader) ConfigureHost(filterFilePath string) error {
	cmd := exec.Command(sL.Path, "--configure", filterFilePath)
	cmd.Env = os.Environ()
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

func (sL *ScopeLoader) ConfigureContainer(filterFilePath string, cpid int) error {
	cmd := exec.Command(sL.Path, "--configure", filterFilePath, "--namespace", strconv.Itoa(cpid))
	cmd.Env = os.Environ()
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

func (sL *ScopeLoader) ServiceHost(serviceName string) error {
	cmd := exec.Command(sL.Path, "--service", serviceName)
	cmd.Env = os.Environ()
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

func (sL *ScopeLoader) ServiceContainer(serviceName string, cpid int) error {
	cmd := exec.Command(sL.Path, "--service", serviceName, "--namespace", strconv.Itoa(cpid))
	cmd.Env = os.Environ()
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

func (sL *ScopeLoader) Patch(libraryPath string) error {
	cmd := exec.Command(sL.Path, "--patch", libraryPath)
	cmd.Env = os.Environ()
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

func (sL *ScopeLoader) Run(args []string, env []string) error {
	args = append([]string{"ldscope"}, args...)
	return syscall.Exec(sL.Path, args, env)
}

func (sL *ScopeLoader) RunSubProc(args []string, env []string) error {
	cmd := exec.Command(sL.Path, args...)
	cmd.Env = env
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

func (sL *ScopeLoader) Attach(args []string, env []string) error {
	args = append([]string{"--attach"}, args...)
	return sL.Run(args, env)
}

func (sL *ScopeLoader) AttachSubProc(args []string, env []string) error {
	args = append([]string{"--attach"}, args...)
	return sL.RunSubProc(args, env)
}

package run

import (
	"os"
	"os/exec"
	"path"
	"strconv"
)

// TODO move code below to separate package since it is not related with config at all
func (rc *Config) Patch(scope_path string) {
	cmd := exec.Command(path.Join(scope_path, "ldscope"), "-p", path.Join(scope_path, "libscope.so"))
	cmd.Env = os.Environ()
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	_ = cmd.Run()
}

func (rc *Config) SetupContainer(cpid int) error {
	cmd := exec.Command(ldscopePath(), "-s", strconv.Itoa(cpid))
	cmd.Env = os.Environ()
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

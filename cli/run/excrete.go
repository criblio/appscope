package run

import (
	"os"
	"os/exec"
	"path"
)

func (rc *Config) Patch(scope_path string) {
	cmd := exec.Command(path.Join(scope_path, "ldscope"), "-p", path.Join(scope_path, "libscope.so"))
	cmd.Env = os.Environ()
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	_ = cmd.Run()
}

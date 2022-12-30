package ipc

import (
	"os"
	"os/exec"
	"syscall"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestCompareSelfIpcNs(t *testing.T) {
	selfPid := os.Getpid()
	sameIPC, err := ipcNsIsSame(selfPid)
	assert.NoError(t, err)
	assert.True(t, sameIPC)
}

func TestCompareDiffIpcNs(t *testing.T) {
	cmd := exec.Command("sh")
	cmd.SysProcAttr = &syscall.SysProcAttr{
		Cloneflags: syscall.CLONE_NEWIPC | syscall.CLONE_NEWUSER,
	}
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	err := cmd.Start()
	assert.NoError(t, err)
	sameIPC, err := ipcNsIsSame(cmd.Process.Pid)
	assert.NoError(t, err)
	assert.False(t, sameIPC)
	err = cmd.Wait()
	assert.NoError(t, err)
}

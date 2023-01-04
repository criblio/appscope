package ipc

import (
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestCompareSelfIpcNs(t *testing.T) {
	selfPid := os.Getpid()
	sameIPC, err := ipcNsIsSame(IpcPidCtx{Pid: selfPid})
	assert.NoError(t, err)
	assert.True(t, sameIPC)
}

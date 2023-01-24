package coredump

import (
	"encoding/json"
	"errors"

	"github.com/criblio/scope/ipc"
)

var errCoreDump = errors.New("error coredump command")

// CoreDumpProcess triggers the core dump in scoped process
func CoreDumpProcess(pidCtx ipc.IpcPidCtx) (string, error) {

	cmdCoreDump := ipc.CmdCoreDump{}
	resp, err := cmdCoreDump.Request(pidCtx)
	if err != nil {
		return "", err
	}

	err = cmdCoreDump.UnmarshalResp(resp.ResponseScopeMsgData)
	if err != nil {
		return "", err
	}
	if resp.MetaMsgStatus != ipc.ResponseOK || *cmdCoreDump.Response.Status != ipc.ResponseOK {
		return "", errCoreDump
	}

	coreDumpStatus, err := json.MarshalIndent(cmdCoreDump.Response, "", "   ")
	if err != nil {
		return "", err
	}

	return string(coreDumpStatus), nil
}

package inspect

import (
	"encoding/json"

	"github.com/criblio/scope/ipc"
)

// InspectScopeCfg returns the configuration of scoped process
func InspectScopeCfg(pidCtx ipc.IpcPidCtx) (string, error) {
	cmd := ipc.CmdGetScopeCfg{}
	respData, err := cmd.Request(pidCtx)
	if err != nil {
		return "", err
	}
	err = cmd.UnmarshalResp(respData.ResponseScopeMsgData)
	if err != nil {
		return "", err
	}

	marshalToPrint, err := json.MarshalIndent(cmd.Response.Cfg.Current, "", "   ")
	if err != nil {
		return "", err
	}

	return string(marshalToPrint), nil
}

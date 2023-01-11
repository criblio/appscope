package inspect

import (
	"encoding/json"
	"errors"

	"github.com/criblio/scope/ipc"
)

var errInspectCfg = errors.New("error inspect cfg")

// InspectScopeCfg returns the configuration of scoped process
func InspectScopeCfg(pidCtx ipc.IpcPidCtx) (string, error) {

	cmd := ipc.CmdGetScopeCfg{}
	resp, err := cmd.Request(pidCtx)
	if err != nil {
		return "", err
	}

	err = cmd.UnmarshalResp(resp.ResponseScopeMsgData)
	if err != nil {
		return "", err
	}
	if resp.MetaMsgStatus != ipc.ResponseOK || *cmd.Response.Status != ipc.ResponseOK {
		return "", errInspectCfg
	}
	marshalToPrint, err := json.MarshalIndent(cmd.Response.Cfg.Current, "", "   ")
	if err != nil {
		return "", err
	}
	return string(marshalToPrint), nil
}

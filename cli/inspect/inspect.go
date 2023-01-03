package inspect

import (
	"bytes"
	"encoding/json"
	"errors"

	"github.com/criblio/scope/ipc"
)

var errInspectCfg = errors.New("error inspect cfg")

// InspectScopeCfg returns the configuration of scoped process
func InspectScopeCfg(pidCtx ipc.IpcPidCtx) (string, error) {
	var prettyJSON bytes.Buffer

	cmd := ipc.CmdGetScopeCfg{}
	resp, err := cmd.Request(pidCtx)
	if err != nil {
		return "", err
	}

	// TODO fix me: correct parameters type
	// err = cmd.UnmarshalResp(resp.ResponseScopeMsgData)
	// if err != nil {
	// 	return "", err
	// }
	// if resp.MetaMsgStatus != ipc.ResponseOK || cmd.Response.Status != ipc.ResponseOK {
	// 	return errInspectCfg
	// }
	// marshalToPrint, err := json.MarshalIndent(cmd.Response.Cfg, "", "   ")
	// if err != nil {
	// 	return "", err
	// }
	// return string(marshalToPrint), nil

	if resp.MetaMsgStatus != ipc.ResponseOK {
		return "", errInspectCfg
	}

	if err := json.Indent(&prettyJSON, []byte(resp.ResponseScopeMsgData), "", "    "); err != nil {
		return "", err
	}
	return prettyJSON.String(), nil
}

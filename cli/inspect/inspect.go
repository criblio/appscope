package inspect

import (
	"encoding/json"
	"errors"

	"github.com/criblio/scope/ipc"
)

var errInspectCfg = errors.New("error inspect cfg")

type inspectOutput struct {
	Cfg  ipc.ScopeGetCfgResponseCfg `mapstructure:"cfg" json:"cfg" yaml:"cfg"`
	Desc ipc.ScopeInterfaceDesc     `mapstructure:"interfaces" json:"interfaces" yaml:"interfaces"`
}

// InspectScopeCfg returns the configuration of scoped process
func InspectScopeCfg(pidCtx ipc.IpcPidCtx) (string, error) {

	cmdGetCfg := ipc.CmdGetScopeCfg{}
	resp, err := cmdGetCfg.Request(pidCtx)
	if err != nil {
		return "", err
	}

	err = cmdGetCfg.UnmarshalResp(resp.ResponseScopeMsgData)
	if err != nil {
		return "", err
	}
	if resp.MetaMsgStatus != ipc.ResponseOK || *cmdGetCfg.Response.Status != ipc.ResponseOK {
		return "", errInspectCfg
	}

	cmdGetTransportStatus := ipc.CmdGetTransportStatus{}
	respTr, err := cmdGetTransportStatus.Request(pidCtx)
	if err != nil {
		return "", err
	}

	err = cmdGetTransportStatus.UnmarshalResp(respTr.ResponseScopeMsgData)
	if err != nil {
		return "", err
	}

	if respTr.MetaMsgStatus != ipc.ResponseOK || *cmdGetTransportStatus.Response.Status != ipc.ResponseOK {
		return "", errInspectCfg
	}

	summary := inspectOutput{
		Cfg:  cmdGetCfg.Response.Cfg,
		Desc: cmdGetTransportStatus.Response.Interfaces,
	}

	sumPrint, err := json.MarshalIndent(summary, "", "   ")
	if err != nil {
		return "", err
	}

	return string(sumPrint), nil
}

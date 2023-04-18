package inspect

import (
	"encoding/json"
	"errors"

	"github.com/criblio/scope/ipc"
)

var errInspectCfg = errors.New("error inspect cfg")

type InspectOutput struct {
	Cfg  ipc.ScopeGetCfgResponseCfg `mapstructure:"cfg" json:"cfg" yaml:"cfg"`
	Desc ipc.ScopeInterfaceDesc     `mapstructure:"interfaces" json:"interfaces" yaml:"interfaces"`
}

// InspectProcess returns the configuratioutn and status of scoped process
func InspectProcess(pidCtx ipc.IpcPidCtx) (InspectOutput, string, error) {
	var iout InspectOutput

	cmdGetCfg := ipc.CmdGetScopeCfg{}
	resp, err := cmdGetCfg.Request(pidCtx)
	if err != nil {
		return iout, "", err
	}

	err = cmdGetCfg.UnmarshalResp(resp.ResponseScopeMsgData)
	if err != nil {
		return iout, "", err
	}
	if resp.MetaMsgStatus != ipc.ResponseOK || *cmdGetCfg.Response.Status != ipc.ResponseOK {
		return iout, "", errInspectCfg
	}

	cmdGetTransportStatus := ipc.CmdGetTransportStatus{}
	respTr, err := cmdGetTransportStatus.Request(pidCtx)
	if err != nil {
		return iout, "", err
	}

	err = cmdGetTransportStatus.UnmarshalResp(respTr.ResponseScopeMsgData)
	if err != nil {
		return iout, "", err
	}

	if respTr.MetaMsgStatus != ipc.ResponseOK || *cmdGetTransportStatus.Response.Status != ipc.ResponseOK {
		return iout, "", errInspectCfg
	}

	iout = InspectOutput{
		Cfg:  cmdGetCfg.Response.Cfg,
		Desc: cmdGetTransportStatus.Response.Interfaces,
	}

	sumPrint, err := json.MarshalIndent(iout, "", "   ")
	if err != nil {
		return iout, "", err
	}

	return iout, string(sumPrint), nil
}

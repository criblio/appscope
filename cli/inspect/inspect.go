package inspect

import (
	"encoding/json"
	"errors"

	"github.com/criblio/scope/ipc"
)

var errInspectCfg = errors.New("error inspect cfg")

// procDetails describes details of process
type procDetails struct {
	// Pid
	Pid int `mapstructure:"pid" json:"pid" yaml:"pid"`
	// UUID
	Uuid string `mapstructure:"uuid" json:"uuid" yaml:"uuid"`
	// Id
	Id string `mapstructure:"id" json:"id" yaml:"id"`
	// Machine id
	MachineId string `mapstructure:"machine_id" json:"machine_id" yaml:"machine_id"`
}

type inspectOutput struct {
	Cfg     ipc.ScopeGetCfgResponseCfg `mapstructure:"cfg" json:"cfg" yaml:"cfg"`
	Desc    ipc.ScopeInterfaceDesc     `mapstructure:"interfaces" json:"interfaces" yaml:"interfaces"`
	Process procDetails                `mapstructure:"process" json:"process" yaml:"process"`
}

// InspectProcess returns the configuration and status of scoped process
func InspectProcess(pidCtx ipc.IpcPidCtx) (string, error) {

	// Get configuration
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

	// Get transport status
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

	// Get process details
	cmdGetProcDetails := ipc.CmdGetProcessDetails{}
	respPd, err := cmdGetProcDetails.Request(pidCtx)
	if err != nil {
		return "", err
	}

	err = cmdGetProcDetails.UnmarshalResp(respPd.ResponseScopeMsgData)
	if err != nil {
		return "", err
	}

	if respPd.MetaMsgStatus != ipc.ResponseOK || *cmdGetProcDetails.Response.Status != ipc.ResponseOK {
		return "", errInspectCfg
	}

	summary := inspectOutput{
		Cfg:  cmdGetCfg.Response.Cfg,
		Desc: cmdGetTransportStatus.Response.Interfaces,
		Process: procDetails{
			Pid:       cmdGetProcDetails.Response.Pid,
			Uuid:      cmdGetProcDetails.Response.Uuid,
			Id:        cmdGetProcDetails.Response.Id,
			MachineId: cmdGetProcDetails.Response.MachineId,
		},
	}

	sumPrint, err := json.MarshalIndent(summary, "", "   ")
	if err != nil {
		return "", err
	}

	return string(sumPrint), nil
}

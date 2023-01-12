package inspect

import (
	"encoding/json"
	"errors"

	"github.com/criblio/scope/ipc"
)

var errInspectCfg = errors.New("error inspect cfg")

type inspectOutput struct {
	Cfg  ipc.ScopeGetCfgResponseCfg `mapstructure:"cfg" json:"cfg" yaml:"cfg"`
	Desc ipc.ChannelDesc            `mapstructure:"channels" json:"channels" yaml:"channels"`
}

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

	cmdTr := ipc.CmdGetTransportStatus{}
	respTr, err := cmdTr.Request(pidCtx)
	if err != nil {
		return "", err
	}

	err = cmdTr.UnmarshalResp(respTr.ResponseScopeMsgData)
	if err != nil {
		return "", err
	}

	if respTr.MetaMsgStatus != ipc.ResponseOK || *cmdTr.Response.Status != ipc.ResponseOK {
		return "", errInspectCfg
	}

	summary := inspectOutput{
		Cfg:  cmd.Response.Cfg,
		Desc: cmdTr.Response.Channels,
	}

	sumPrint, err := json.MarshalIndent(summary, "", "   ")
	if err != nil {
		return "", err
	}

	return string(sumPrint), nil
}

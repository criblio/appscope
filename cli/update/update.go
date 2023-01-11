package update

import (
	"errors"
	"os"

	"github.com/criblio/scope/ipc"
)

var errSettingCfg = errors.New("error setting cfg")

// UpdateScopeCfg updates the configuration of scoped process
func UpdateScopeCfg(pidCtx ipc.IpcPidCtx, confFile string) error {
	content, err := os.ReadFile(confFile)
	if err != nil {
		return err
	}
	cmd := ipc.CmdSetScopeCfg{CfgData: content}
	resp, err := cmd.Request(pidCtx)
	if err != nil {
		return err
	}
	err = cmd.UnmarshalResp(resp.ResponseScopeMsgData)
	if err != nil {
		return err
	}

	if resp.MetaMsgStatus != ipc.ResponseOK || *cmd.Response.Status != ipc.ResponseOK {
		return errSettingCfg
	}

	return nil
}

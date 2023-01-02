package update

import (
	"os"

	"github.com/criblio/scope/ipc"
)

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

	return nil
}

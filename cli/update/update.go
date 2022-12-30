package update

import (
	"os"

	"github.com/criblio/scope/ipc"
)

// UpdateScopeCfg updates the configuration of scoped process
func UpdateScopeCfg(pid int, confFile string) error {
	content, err := os.ReadFile(confFile)
	if err != nil {
		return err
	}
	cmd := ipc.CmdSetScopeCfg{CfgData: content}

	resp, err := cmd.Request(pid)
	if err != nil {
		return err
	}
	err = cmd.UnmarshalResp(resp.ResponseScopeMsgData)
	if err != nil {
		return err
	}

	return nil
}

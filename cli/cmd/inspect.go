package cmd

import (
	"errors"
	"fmt"
	"strconv"

	"github.com/criblio/scope/inspect"
	"github.com/criblio/scope/internal"
	"github.com/criblio/scope/ipc"
	"github.com/criblio/scope/util"
	"github.com/spf13/cobra"
)

var pidCtx *ipc.IpcPidCtx = &ipc.IpcPidCtx{}

// inspectCmd represents the inspect command
var inspectCmd = &cobra.Command{
	Use:     "inspect",
	Short:   "Return information on scoped process",
	Long:    `Return information on scoped process identified by PID.`,
	Example: `scope inspect 1000`,
	Args:    cobra.ExactArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		internal.InitConfig()
		// Nice message for non-adminstrators
		err := util.UserVerifyRootPerm()
		if errors.Is(err, util.ErrGetCurrentUser) {
			util.ErrAndExit("Unable to get current user: %v", err)
		}
		if errors.Is(err, util.ErrMissingAdmPriv) {
			fmt.Println("INFO: Run as root (or via sudo) to get info from all processes")
		}

		pid, err := strconv.Atoi(args[0])
		if err != nil {
			util.ErrAndExit("Convert PID fails: %v", err)
		}
		pidCtx.Pid = pid
		cfg, err := inspect.InspectScopeCfg(*pidCtx)
		if err != nil {
			util.ErrAndExit("Inspect PID fails: %v", err)
		}
		fmt.Println(cfg)
	},
}

func init() {
	ipcCmdFlags(inspectCmd, pidCtx)
	RootCmd.AddCommand(inspectCmd)
}

package cmd

import (
	"errors"
	"fmt"
	"strconv"

	"github.com/criblio/scope/coredump"

	"github.com/criblio/scope/internal"
	"github.com/criblio/scope/ipc"
	"github.com/criblio/scope/util"
	"github.com/spf13/cobra"
)

var corePidCtx *ipc.IpcPidCtx = &ipc.IpcPidCtx{}

// coredumpCmd represents the coredump command
var coredumpCmd = &cobra.Command{
	Use:     "coredump",
	Short:   "Force core dump on scoped process",
	Long:    `Force core dump on scoped process`,
	Example: `scope coredump 1000`,
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
		corePidCtx.Pid = pid
		cfg, err := coredump.CoreDumpProcess(*corePidCtx)
		if err != nil {
			util.ErrAndExit("Inspect PID fails: %v", err)
		}
		fmt.Println(cfg)
	},
}

func init() {
	ipcCmdFlags(coredumpCmd, corePidCtx)
	RootCmd.AddCommand(coredumpCmd)
}

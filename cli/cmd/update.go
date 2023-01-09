package cmd

import (
	"errors"
	"fmt"
	"strconv"

	"github.com/criblio/scope/internal"
	"github.com/criblio/scope/update"
	"github.com/criblio/scope/util"
	"github.com/spf13/cobra"
)

var cfgPath string

// updateCmd represents the info command
var updateCmd = &cobra.Command{
	Use:     "update",
	Short:   "Updates configuration of scoped process",
	Long:    `Updates configuration of scoped process identified by PID.`,
	Example: `scope update 1000 --config test_cfg.yml`,
	Args:    cobra.ExactArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		if !util.CheckFileExists(cfgPath) {
			util.ErrAndExit("Configuration file: %s does not exist", cfgPath)
		}
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

		err = update.UpdateScopeCfg(*pidCtx, cfgPath)
		if err != nil {
			util.ErrAndExit("Update Scope configuration fails: %v", err)
		}
		fmt.Println("Update Scope configuration success.")
	},
}

func init() {
	ipcCmdFlags(updateCmd, pidCtx)
	updateCmd.Flags().StringVarP(&cfgPath, "config", "c", "", "Path to configuration file")
	updateCmd.MarkFlagRequired("config")
	RootCmd.AddCommand(updateCmd)
}

package cmd

import (
	"errors"
	"fmt"
	"os"
	"strconv"
	"time"

	"github.com/criblio/scope/inspect"
	"github.com/criblio/scope/internal"
	"github.com/criblio/scope/ipc"
	"github.com/criblio/scope/libscope"
	"github.com/criblio/scope/update"
	"github.com/criblio/scope/util"
	"github.com/spf13/cobra"
	"gopkg.in/yaml.v2"
)

// updateCmd represents the info command
var updateCmd = &cobra.Command{
	Use:     "update",
	Short:   "Updates the configuration of a scoped process",
	Long:    `Updates the configuration of a scoped process identified by PID.`,
	Example: `scope update 1000 --config test_cfg.yml`,
	Args:    cobra.ExactArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		internal.InitConfig()
		prefix, _ := cmd.Flags().GetString("prefix")
		fetch, _ := cmd.Flags().GetBool("fetch")
		cfgPath, _ := cmd.Flags().GetString("config")

		// Nice message for non-adminstrators
		err := util.UserVerifyRootPerm()
		if errors.Is(err, util.ErrGetCurrentUser) {
			util.ErrAndExit("Unable to get current user: %v", err)
		}
		if errors.Is(err, util.ErrMissingAdmPriv) {
			fmt.Println("INFO: Run as root (or via sudo) to get info from all processes")
		}

		var cfgBytes []byte
		if cfgPath != "" {
			// User specified a path to a config with --config
			if !util.CheckFileExists(cfgPath) {
				util.ErrAndExit("Configuration file: %s does not exist", cfgPath)
			}
			cfgBytes, err = os.ReadFile(cfgPath)
			if err != nil {
				util.ErrAndExit("Unable to read bytes from config path", err)
			}
		} else {
			// User did not specigy a path to a config with --config. Try to read from StdIn
			var scopeCfg libscope.ScopeConfig
			if scopeCfg, err = update.GetCfgStdIn(); err != nil {
				util.ErrAndExit("Unable to parse config from stdin", err)
			}
			if cfgBytes, err = yaml.Marshal(scopeCfg); err != nil {
				util.ErrAndExit("Unable to marshal scope config into byte array", err)
			}
		}

		pid, err := strconv.Atoi(args[0])
		if err != nil {
			util.ErrAndExit("error parsing PID argument")
		}

		status, _ := util.PidScopeLibInMaps(pid)
		if !status {
			util.ErrAndExit("Unable to communicate with %v - process is not scoped", pid)
		}

		pidCtx := &ipc.IpcPidCtx{
			PrefixPath: prefix,
			Pid:        pid,
		}

		err = update.UpdateScopeCfg(*pidCtx, cfgBytes)
		if err != nil {
			util.ErrAndExit("Update Scope configuration fails: %v", err)
		}
		fmt.Println("Update Scope configuration success.")

		if fetch {
			time.Sleep(2 * time.Second)
			_, cfg, err := inspect.InspectProcess(*pidCtx)
			if err != nil {
				util.ErrAndExit("Inspect PID fails: %v", err)
			}

			fmt.Println(cfg)
		}
	},
}

func init() {
	updateCmd.Flags().BoolP("fetch", "f", false, "Inspect the process after the update is complete")
	updateCmd.Flags().StringP("prefix", "p", "", "Prefix to proc filesystem")
	updateCmd.Flags().BoolP("json", "j", false, "Output as newline delimited JSON")
	updateCmd.Flags().StringP("config", "c", "", "Path to configuration file")
	RootCmd.AddCommand(updateCmd)
}

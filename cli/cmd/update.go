package cmd

import (
	"encoding/json"
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
	Use:   "update",
	Short: "Updates the configuration of a scoped process",
	Long:  `Updates the configuration of a scoped process identified by PID.`,
	Example: `  scope update 1000 --config scope_cfg.yml
  scope update 1000 < scope_cfg.yml
  scope update 1000 --json < scope_cfg.yml
  scope update 1000 --rootdir /path/to/host/root --config scope_cfg.yml
  scope update 1000 --rootdir /path/to/host/root/proc/<hostpid>/root < scope_cfg.yml`,
	Args: cobra.ExactArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		internal.InitConfig()
		rootdir, _ := cmd.Flags().GetString("rootdir")
		inspectFlag, _ := cmd.Flags().GetBool("inspect")
		cfgPath, _ := cmd.Flags().GetString("config")
		jsonOut, _ := cmd.Flags().GetBool("json")

		// Nice message for non-adminstrators
		err := util.UserVerifyRootPerm()
		if errors.Is(err, util.ErrGetCurrentUser) {
			util.ErrAndExit("Unable to get current user: %v", err)
		}
		admin := true
		if errors.Is(err, util.ErrMissingAdmPriv) {
			admin = false
		}

		var cfgBytes []byte
		if cfgPath != "" {
			// User specified a path to a config with --config
			if !util.CheckFileExists(cfgPath) {
				util.ErrAndExit("Configuration file: %s does not exist", cfgPath)
			}
			cfgBytes, err = os.ReadFile(cfgPath)
			if err != nil {
				util.ErrAndExit("Unable to read bytes from config path: %v", err)
			}
		} else {
			// User did not specify a path to a config with --config. Try to read from StdIn
			var scopeCfg libscope.ScopeConfig
			if scopeCfg, err = update.GetCfgStdIn(); err != nil {
				util.ErrAndExit("Unable to parse config from stdin: %v", err)
			}
			if cfgBytes, err = yaml.Marshal(scopeCfg); err != nil {
				util.ErrAndExit("Unable to marshal scope config into byte array: %v", err)
			}
		}

		pid, err := strconv.Atoi(args[0])
		if err != nil {
			util.ErrAndExit("error parsing PID argument")
		}

		status, _ := util.PidScopeLibInMaps(rootdir, pid)
		if !status {
			if !admin {
				util.Warn("INFO: Run as root (or via sudo) to interact with all processes")
			}
			util.ErrAndExit("Unable to communicate with %v - process is not scoped", pid)
		}

		pidCtx := &ipc.IpcPidCtx{
			PrefixPath: rootdir,
			Pid:        pid,
		}

		err = update.UpdateScopeCfg(*pidCtx, cfgBytes)
		if err != nil {
			if !admin {
				util.Warn("INFO: Run as root (or via sudo) to interact with all processes")
			}
			util.ErrAndExit("Update Scope configuration fails: %v", err)
		}
		util.Warn("Update Scope configuration success.")

		if inspectFlag {
			time.Sleep(2 * time.Second)
			iout, _, err := inspect.InspectProcess(*pidCtx)
			if err != nil {
				if !admin {
					util.Warn("INFO: Run as root (or via sudo) to interact with all processes")
				}
				util.ErrAndExit("Inspect PID fails: %v", err)
			}

			if jsonOut {
				// Print the json object without any pretty printing
				cfg, err := json.Marshal(iout)
				if err != nil {
					util.ErrAndExit("Error creating json object: %v", err)
				}
				fmt.Println(string(cfg))
			} else {
				// Print the json in a pretty format
				cfg, err := json.MarshalIndent(iout, "", "   ")
				if err != nil {
					util.ErrAndExit("Error creating json array: %v", err)
				}
				fmt.Println(string(cfg))
			}
		}
	},
}

func init() {
	updateCmd.Flags().BoolP("inspect", "i", false, "Inspect the process after the update is complete")
	updateCmd.Flags().StringP("rootdir", "R", "", "Path to root filesystem of target namespace")
	updateCmd.Flags().BoolP("json", "j", false, "Output as newline delimited JSON")
	updateCmd.Flags().StringP("config", "c", "", "Path to configuration file")
	RootCmd.AddCommand(updateCmd)
}

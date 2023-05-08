package cmd

import (
	"encoding/json"
	"errors"
	"fmt"
	"strconv"

	"github.com/criblio/scope/inspect"
	"github.com/criblio/scope/internal"
	"github.com/criblio/scope/ipc"
	"github.com/criblio/scope/run"
	"github.com/criblio/scope/util"
	"github.com/spf13/cobra"
)

// inspectCmd represents the inspect command
var inspectCmd = &cobra.Command{
	Use:   "inspect",
	Short: "Returns information about scoped process",
	Long:  `Returns information about scoped process identified by PID.`,
	Example: `scope inspect
scope inspect 1000
scope inspect --all`,
	Args: cobra.MaximumNArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		internal.InitConfig()
		all, _ := cmd.Flags().GetBool("all")
		prefix, _ := cmd.Flags().GetString("prefix")
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

		pidCtx := &ipc.IpcPidCtx{
			PrefixPath: prefix,
		}

		if all {
			// Get all scoped processes
			procs, err := util.ProcessesScoped()
			if err != nil {
				if !admin {
					util.Warn("INFO: Run as root (or via sudo) to interact with all processes")
				}
				util.ErrAndExit("Unable to retrieve scoped processes: %v", err)
			}

			// Inspect each of them and store output in an array
			iouts := make([]inspect.InspectOutput, 0)
			for _, proc := range procs {
				pidCtx.Pid = proc.Pid
				iout, _, err := inspect.InspectProcess(*pidCtx)
				if err != nil {
					if !admin {
						util.Warn("INFO: Run as root (or via sudo) to interact with all processes")
					}
					util.Warn("Inspect PID fails: %v", err)
				}
				iouts = append(iouts, iout)
			}

			if jsonOut {
				// Print each json entry on a newline, without any pretty printing
				for _, iout := range iouts {
					cfg, err := json.Marshal(iout)
					if err != nil {
						util.ErrAndExit("Error creating json object: %v", err)
					}
					fmt.Println(string(cfg))
				}
			} else {
				// Print the array, in a pretty format
				cfgs, err := json.MarshalIndent(iouts, "", "   ")
				if err != nil {
					util.ErrAndExit("Error creating json array: %v", err)
				}
				fmt.Println(string(cfgs))
			}

			return
		}

		var pid int
		if len(args) == 0 {
			// If no pid argument was provided
			// Helper menu to allow the user to select a pid to inspect

			if pid, err = run.HandleInputArg("", false, true, false); err != nil {
				if !admin {
					util.Warn("INFO: Run as root (or via sudo) to interact with all processes")
				}
				util.ErrAndExit("No scoped processes to inspect")
			}

		} else {
			// If a user specified a pid as an argument
			pid, err = strconv.Atoi(args[0])
			if err != nil {
				util.ErrAndExit("error parsing PID argument")
			}
		}

		status, _ := util.PidScopeLibInMaps(pid)
		if !status {
			if !admin {
				util.Warn("INFO: Run as root (or via sudo) to interact with all processes")
			}
			util.ErrAndExit("Unable to communicate with %v - process is not scoped", pid)
		}

		pidCtx.Pid = pid
		_, cfg, err := inspect.InspectProcess(*pidCtx)
		if err != nil {
			if !admin {
				util.Warn("INFO: Run as root (or via sudo) to interact with all processes")
			}
			util.ErrAndExit("Inspect PID fails: %v", err)
		}

		fmt.Println(cfg)
	},
}

func init() {
	inspectCmd.Flags().StringP("prefix", "p", "", "Prefix to proc filesystem")
	inspectCmd.Flags().BoolP("json", "j", false, "Output as newline delimited JSON")
	inspectCmd.Flags().BoolP("all", "a", false, "Inspect all processes")
	RootCmd.AddCommand(inspectCmd)
}

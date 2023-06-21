package cmd

import (
	"encoding/json"
	"errors"
	"fmt"

	"github.com/criblio/scope/inspect"
	"github.com/criblio/scope/internal"
	"github.com/criblio/scope/ipc"
	"github.com/criblio/scope/util"
	"github.com/rs/zerolog/log"
	"github.com/spf13/cobra"
)

var (
	errInspectingMultiple = errors.New("at least one error found when inspecting more than 1 process. See logs")
)

// inspectCmd represents the inspect command
var inspectCmd = &cobra.Command{
	Use:   "inspect",
	Short: "Returns information about scoped process",
	Long:  `Returns information about scoped process identified by PID.`,
	Example: `  scope inspect
  scope inspect 1000
  scope inspect --all --json
  scope inspect 1000 --rootdir /path/to/host/root
  scope inspect --all --rootdir /path/to/host/root
  scope inspect --all --rootdir /path/to/host/root/proc/<hostpid>/root`,
	Args: cobra.MaximumNArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		internal.InitConfig()
		all, _ := cmd.Flags().GetBool("all")
		rootdir, _ := cmd.Flags().GetString("rootdir")
		jsonOut, _ := cmd.Flags().GetBool("json")

		// Nice message for non-adminstrators
		err := util.UserVerifyRootPerm()
		if errors.Is(err, util.ErrGetCurrentUser) {
			util.ErrAndExit("Unable to get current user: %v", err)
		}

		if all && len(args) > 0 {
			helpErrAndExit(cmd, "--all flag is mutually exclusive with PID or <process_name>")
		}

		pidCtx := &ipc.IpcPidCtx{
			PrefixPath: rootdir,
		}
		iouts := make([]inspect.InspectOutput, 0)

		id := ""
		if len(args) > 0 {
			id = args[0]
		}

		procs, err := util.HandleInputArg(id, "", rootdir, !all, false, false, false)
		if err != nil {
			return err
		}

		if len(procs) == 0 {
			return errNoScopedProcs
		}
		if len(procs) == 1 {
			pidCtx.Pid = procs[0].Pid
			iout, _, err := inspect.InspectProcess(*pidCtx)
			if err != nil {
				return err
			}
			iouts = append(iouts, iout)
		} else { // len(procs) > 1
			errors := false
			for _, proc := range procs {
				pidCtx.Pid = proc.Pid
				iout, _, err := inspect.InspectProcess(*pidCtx)
				if err != nil {
					log.Error().Err(err)
					errors = true
				}
				iouts = append(iouts, iout)
			}
			if errors {
				return errInspectingMultiple
			}
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
			var cfgs []byte
			// Print the array, in a pretty format
			if len(iouts) == 1 {
				// No need to create an array
				cfgs, err = json.MarshalIndent(iouts[0], "", "   ")
			} else {
				cfgs, err = json.MarshalIndent(iouts, "", "   ")
			}
			if err != nil {
				util.ErrAndExit("Error creating json array: %v", err)
			}
			fmt.Println(string(cfgs))
		}

		return nil
	},
}

func init() {
	inspectCmd.Flags().StringP("rootdir", "R", "", "Path to root filesystem of target namespace")
	inspectCmd.Flags().BoolP("json", "j", false, "Output as newline delimited JSON")
	inspectCmd.Flags().BoolP("all", "a", false, "Inspect all processes")
	RootCmd.AddCommand(inspectCmd)
}

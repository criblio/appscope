package cmd

import (
	"errors"
	"fmt"

	"github.com/criblio/scope/internal"
	"github.com/criblio/scope/util"
	"github.com/rs/zerolog/log"
	"github.com/spf13/cobra"
)

var (
	errNoScopedProcs     = errors.New("no scoped processes found")
	errDetachingMultiple = errors.New("at least one error found when detaching from more than 1 process. See logs")
)

/* Args Matrix (X disallows)
 *             all
 * all          -
 */

// detachCmd represents the detach command
var detachCmd = &cobra.Command{
	Use:   "detach [flags] PID | <process_name>",
	Short: "Unscope a currently-running process",
	Long:  `Unscopes a currently-running process identified by PID or <process_name>.`,
	Example: `  scope detach 1000
  scope detach firefox
  scope detach --all
  scope detach 1000 --rootdir /path/to/host/root
  scope detach --rootdir /path/to/host/root
  scope detach --all --rootdir /path/to/host/root/proc/<hostpid>/root`,
	Args: cobra.MaximumNArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		internal.InitConfig()
		all, _ := cmd.Flags().GetBool("all")
		rc.Rootdir, _ = cmd.Flags().GetString("rootdir")

		if all && len(args) > 0 {
			helpErrAndExit(cmd, "--all flag is mutually exclusive with PID or <process_name>")
		}

		id := ""
		if len(args) > 0 {
			id = args[0]
		}

		procs, err := util.HandleInputArg(id, rc.Rootdir, !all, true, false, false)
		if err != nil {
			return err
		}

		if len(procs) == 0 {
			return errNoScopedProcs
		}
		if len(procs) == 1 {
			return rc.Detach(procs[0].Pid)
		}
		// len(procs) is > 1
		if !util.Confirm(fmt.Sprintf("Are your sure you want to detach from all of these processes?")) {
			fmt.Println("info: canceled")
			return nil
		}

		errors := false
		for _, proc := range procs {
			if err = rc.Detach(proc.Pid); err != nil {
				log.Error().Err(err)
				errors = true
			}
		}
		if errors {
			return errDetachingMultiple
		}

		return nil
	},
}

func init() {
	detachCmd.Flags().BoolP("all", "a", false, "Detach from all processes")
	detachCmd.Flags().StringP("rootdir", "R", "", "Path to root filesystem of target namespace")
	RootCmd.AddCommand(detachCmd)
}

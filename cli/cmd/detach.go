package cmd

import (
	"github.com/criblio/scope/internal"
	"github.com/spf13/cobra"
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

		id := ""
		if all {
			if len(args) != 0 {
				helpErrAndExit(cmd, "--all flag is mutually exclusive with PID or <process_name>")
			}
			return rc.Detach(id, false, true)
		}

		if len(args) > 0 {
			id = args[0]
		}
		return rc.Detach(id, true, true)
	},
}

func init() {
	detachCmd.Flags().BoolP("all", "a", false, "Detach from all processes")
	detachCmd.Flags().StringP("rootdir", "R", "", "Path to root filesystem of target namespace")
	RootCmd.AddCommand(detachCmd)
}

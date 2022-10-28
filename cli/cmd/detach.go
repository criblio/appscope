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
	Example: `scope detach 1000
scope detach firefox
scope detach --all`,
	Args: cobra.MaximumNArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		internal.InitConfig()
		all, _ := cmd.Flags().GetBool("all")

		// Disallow bad argument combinations (see Arg Matrix at top of file)
		if !all && len(args) == 0 {
			helpErrAndExit(cmd, "Must specify a pid, process name, or --all")
		}

		if all {
			rc.Subprocess = true
			return rc.DetachAll(args)
		}
		return rc.DetachSingle(args)
	},
}

func init() {
	detachCmd.Flags().BoolP("all", "a", false, "Detach from all processes")
	RootCmd.AddCommand(detachCmd)
}

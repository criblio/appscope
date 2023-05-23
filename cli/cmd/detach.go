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
		rc.Rootdir, _ = cmd.Flags().GetString("rootdir")

		if all {
			if len(args) != 0 {
				helpErrAndExit(cmd, "--all flag is mutual exclusive with PID or <process_name>")
			}
			rc.Subprocess = true
			return rc.DetachAll(true)
		}
		if len(args) == 0 {
			return rc.DetachSingle("")
		}

		return rc.DetachSingle(args[0])
	},
}

func init() {
	detachCmd.Flags().BoolP("all", "a", false, "Detach from all processes")
	detachCmd.Flags().StringP("rootdir", "R", "", "Path to root filesystem of target namespace")
	RootCmd.AddCommand(detachCmd)
}

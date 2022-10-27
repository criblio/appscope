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
	Args: cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		internal.InitConfig()

		all, _ := cmd.Flags().GetBool("all")

		if all {

		}

		return rc.Detach(args)
	},
}

func init() {
	detachCmd.Flags().BoolP("all", "a", false, "Detach from all processes")
	RootCmd.AddCommand(detachCmd)
}

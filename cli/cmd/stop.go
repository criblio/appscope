package cmd

import (
	"fmt"
	"os"

	"github.com/criblio/scope/internal"
	"github.com/criblio/scope/stop"
	"github.com/criblio/scope/util"
	"github.com/spf13/cobra"
)

/* Args Matrix (X disallows)
 *                 force
 * force           -
 */

func getStopUsage() string {
	return `The following actions will be performed:
	- Removal of /etc/ld.so.preload contents
	- Removal of the rules file from /usr/lib/appscope/scope_rules
	- Detach from all currently scoped processes

The command does not uninstall scope or libscope from /usr/lib/appscope or /tmp/appscope
or remove any service configurations`
}

// stopCmd represents the stop command
var stopCmd = &cobra.Command{
	Use:   "stop",
	Short: "Stop scoping all scoped processes and services",
	Long: `Stop scoping all processes and services on the host and in all relevant containers.

` + getStopUsage(),
	Example: `  scope stop`,
	Args:    cobra.NoArgs,
	Run: func(cmd *cobra.Command, args []string) {
		internal.InitConfig()
		rc.Rootdir, _ = cmd.Flags().GetString("rootdir")

		force, _ := cmd.Flags().GetBool("force")
		if !force {
			fmt.Println(getStopUsage())
			fmt.Println("\nIf you wish to proceed, run again with the -f flag.")
			os.Exit(0)
		}
		if err := stop.Stop(rc); err != nil {
			util.ErrAndExit("Exiting due to stop failure: %v", err)
		}
	},
}

func init() {
	stopCmd.Flags().StringP("rootdir", "R", "", "Path to root filesystem of target namespace")
	stopCmd.Flags().BoolP("force", "f", false, "Use this flag when you're sure you want to run scope stop")
	RootCmd.AddCommand(stopCmd)
}

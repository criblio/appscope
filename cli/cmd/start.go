package cmd

import (
	"fmt"

	"github.com/criblio/scope/util"
	"github.com/spf13/cobra"
)

/* Args Matrix (X disallows)
 *                 force
 * force           -
 */

const startUsage string = `The following actions will be performed on the host and in all relevant containers:
- Extraction of libscope.so to /tmp/libscope.so 
- Extraction of the filter input to /tmp/scope_filter.yml
- Attach to all existing "allowed" processes defined in the filter file

If you run this command with administrator privileges, the following will be performed on the host and in all relevant containers:
- Extraction of libscope.so to /usr/lib/appscope/ 
- Extraction of the filter input to /usr/lib/appscope/
- Attach to all existing "allowed" processes defined in the filter file
- Install etc/profile.d/scope.sh script to preload /usr/lib/appscope/libscope.so
- Modify the relevant service configurations to preload /usr/lib/appscope/libscope.so`

// startCmd represents the start command
var startCmd = &cobra.Command{
	Use:   "start",
	Short: "Start scoping a filtered selection of processes and services",
	Long: `Start scoping a filtered selection of processes and services on the host and in all relevant containers.

` + startUsage,
	Example: `  scope start example_filter.yml`,
	Args:    cobra.ExactArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		force, _ := cmd.Flags().GetBool("force")
		if !force {
			fmt.Printf(startUsage)
			if !util.Confirm("\n\nAre you sure you want to proceed?") {
				util.ErrAndExit("Exiting due to cancelled start command")
			}
		}
		if err := run.Start(args[0]); err != nil {
			util.ErrAndExit("Exiting due to start failure")
		}
	},
}

func init() {
	startCmd.Flags().Bool("force", false, "Bypass confirmation prompt")

	RootCmd.AddCommand(startCmd)
}

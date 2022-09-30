package cmd

import (
	"fmt"
	"os"

	"github.com/criblio/scope/start"
	"github.com/criblio/scope/util"
	"github.com/spf13/cobra"
)

/* Args Matrix (X disallows)
 *                 force
 * force           -
 */

const startUsage string = `The following actions will be performed on the host and in all relevant containers:
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
	Example: `  scope start < example_filter.yml
  cat example_filter.json | scope start`,
	Args: cobra.NoArgs,
	Run: func(cmd *cobra.Command, args []string) {
		force, _ := cmd.Flags().GetBool("force")
		if !force {
			fmt.Printf(startUsage)
			fmt.Println("\n\nIf you wish to proceed, run again with the -f flag.")
			os.Exit(0)
		}
		if err := start.Start(); err != nil {
			util.ErrAndExit("Exiting due to start failure")
		}
	},
}

func init() {
	startCmd.Flags().BoolP("force", "f", false, "Use this flag when you're sure you want to run scope start")

	RootCmd.AddCommand(startCmd)
}

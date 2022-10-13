package cmd

import (
	"fmt"
	"os"

	"github.com/criblio/scope/internal"
	"github.com/criblio/scope/start"
	"github.com/criblio/scope/util"
	"github.com/spf13/cobra"
)

/* Args Matrix (X disallows)
 *                 force
 * force           -
 */

// TODO: Handle start usage depending on version value ? when dev we will use /tmp
func getStartUsage(version string) string {
	return fmt.Sprintf(`The following actions will be performed on the host and in all relevant containers:
	- Extraction of libscope.so to /usr/lib/appscope/%s/ or /tmp/appscope/%s/
	- Extraction of the filter input to /usr/lib/appscope/scope_filter or /tmp/appscope/scope_filter
	- Attach to all existing "allowed" processes defined in the filter file
	- Install etc/profile.d/scope.sh script to preload /usr/lib/appscope/%s/libscope.so or /tmp/appscope/%s/libscope.so
	- Modify the relevant service configurations to preload /usr/lib/appscope/%s/libscope.so or /tmp/appscope/%s/libscope.so`, version, version, version, version, version, version)
}

// startCmd represents the start command
var startCmd = &cobra.Command{
	Use:   "start",
	Short: "Start scoping a filtered selection of processes and services",
	Long: `Start scoping a filtered selection of processes and services on the host and in all relevant containers.

` + getStartUsage("<version>"),
	Example: `  scope start < example_filter.yml
  cat example_filter.json | scope start`,
	Args: cobra.NoArgs,
	Run: func(cmd *cobra.Command, args []string) {
		internal.InitConfig()

		force, _ := cmd.Flags().GetBool("force")
		if !force {
			fmt.Println(getStartUsage(internal.GetNormalizedVersion()))
			fmt.Println("\nIf you wish to proceed, run again with the -f flag.")
			os.Exit(0)
		}
		if err := start.Start(); err != nil {
			util.ErrAndExit("Exiting due to start failure. See the logs for more info.")
		}
	},
}

func init() {
	startCmd.Flags().BoolP("force", "f", false, "Use this flag when you're sure you want to run scope start")

	RootCmd.AddCommand(startCmd)
}

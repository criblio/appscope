package cmd

import (
	"fmt"

	"github.com/spf13/cobra"
)

/* Args Matrix (X disallows)
 */

const usage string = `The following actions will be performed on the host and in all relevant containers:
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

` + usage,
	Example: `  scope start < example_filter.yml
  cat example_filter.json | scope start`,
	Args: cobra.NoArgs,
	Run: func(cmd *cobra.Command, args []string) {
		force, _ := cmd.Flags().GetBool("force")
		if !force {
			fmt.Printf(usage)
		}
		rc.Start(force)
	},
}

func init() {
	startCmd.Flags().Bool("force", false, "Bypass confirmation prompt")

	RootCmd.AddCommand(startCmd)
}

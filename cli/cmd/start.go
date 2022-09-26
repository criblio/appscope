package cmd

import (
	"github.com/spf13/cobra"
)

/* Args Matrix (X disallows)
 */

// startCmd represents the start command
var startCmd = &cobra.Command{
	Use:   "start",
	Short: "Start scoping a filtered selection of processes and services",
	Long: `Start scoping a filtered selection of processes and services on the host and in all relevant containers.

The following actions will be performed on the host and in all relevant containers:
- Extraction of libscope.so to /tmp/libscope.so 
- Extraction of the filter input to /tmp/scope_filter.yml
- Attach to all existing "allowed" processes defined in the filter file

If you choose to persist after reboot, the following will also be performed on the host and in all relevant containers:
- Install etc/profile script to preload /tmp/libscope.so
- Modify the relevant service configurations to preload /tmp/libscope.so`,
	Example: `  scope start < example_filter.yml
  cat example_filter.json | scope start`,
	Args: cobra.NoArgs,
	RunE: func(cmd *cobra.Command, args []string) error {
		return rc.Start()
	},
}

func init() {
	RootCmd.AddCommand(startCmd)
}

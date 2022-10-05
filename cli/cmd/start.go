package cmd

import (
	"github.com/spf13/cobra"
)

// startCmd represents the start command
var startCmd = &cobra.Command{
	Use:   "start",
	Short: "Perform a scope start operation",
	Long: `Perform a scope start operation based on the filter input.

Following actions will be performed:
- extraction of the libscope.so to /tmp/libscope.so on the host and on the containers
- extraction of the filter input to /tmp/scope_filter on the host and on the containers
- setup etc/profile script to use LD_PRELOAD=/tmp/libscope.so on the host and on the containers
- setup the services which meet the allow list conditions on the host and on the containers
- attach to the processes which meet the allow list conditions on the host and on the containers`,
	Example: `
	scope start < example_filter.yml
	cat example_filter.json | scope start
	`,
	Args: cobra.NoArgs,
	RunE: func(cmd *cobra.Command, args []string) error {
		return rc.Start()
	},
}

func init() {
	RootCmd.AddCommand(startCmd)
}

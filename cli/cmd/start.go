package cmd

import (
	"github.com/spf13/cobra"
)

// startCmd represents the start command
var startCmd = &cobra.Command{
	Use:   "start",
	Short: "Start a scoped process list",
	Long:  `Start a scoped process list.`,
	Example: `
	scope start < example_filter.yml
	cat example_filter.json | scope start
	`,
	Args:         cobra.NoArgs,
	SilenceUsage: true,
	RunE: func(cmd *cobra.Command, args []string) error {
		return rc.Start()
	},
}

func init() {
	RootCmd.AddCommand(startCmd)
}

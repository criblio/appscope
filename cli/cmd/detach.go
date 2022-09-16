package cmd

import (
	"github.com/criblio/scope/internal"
	"github.com/spf13/cobra"
)

// detachCmd represents the run command
var detachCmd = &cobra.Command{
	Use:   "detach [flags] PID | <process_name>",
	Short: "Unscope a currently-running process",
	Long:  `Unscopes a currently-running process identified by PID or <process_name>.`,
	Example: `scope detach 1000
scope detach firefox`,
	Args: cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		internal.InitConfig()
		return rc.Detach(args)
	},
}

func init() {
	RootCmd.AddCommand(detachCmd)
}

package cmd

import (
	"github.com/criblio/scope/internal"
	"github.com/criblio/scope/start"
	"github.com/criblio/scope/util"
	"github.com/spf13/cobra"
)

// startCmd represents the start command
var startCmd = &cobra.Command{
	Use:   "start",
	Short: "Install the AppScope library",
	Long: `Install the AppScope library to:
/usr/lib/appscope/<version>/ for release builds, or 
/tmp/appscope/<version>/ for development builds`,
	Example: `scope start`,
	Args:    cobra.NoArgs,
	Run: func(cmd *cobra.Command, args []string) {
		internal.InitConfig()
		prefix, _ := cmd.Flags().GetString("prefix")

		if err := start.Start(prefix); err != nil {
			util.ErrAndExit("Exiting due to start failure: %v", err)
		}
	},
}

func init() {
	startCmd.Flags().StringP("prefix", "p", "", "Prefix to proc filesystem")
	RootCmd.AddCommand(startCmd)
}

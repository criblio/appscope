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
	Example: `scope start
scope start --rootdir /hostfs`,
	Args: cobra.NoArgs,
	Run: func(cmd *cobra.Command, args []string) {
		internal.InitConfig()
		rootdir, _ := cmd.Flags().GetString("rootdir")

		if err := start.Start(rootdir); err != nil {
			util.ErrAndExit("Exiting due to start failure: %v", err)
		}
	},
}

func init() {
	startCmd.Flags().StringP("rootdir", "p", "", "Path to root filesystem of another namespace")
	RootCmd.AddCommand(startCmd)
}

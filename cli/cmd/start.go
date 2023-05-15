package cmd

import (
	"github.com/criblio/scope/internal"
	"github.com/criblio/scope/start"
	"github.com/criblio/scope/util"
	"github.com/rs/zerolog/log"
	"github.com/spf13/cobra"
)

// startCmd represents the start command
var startCmd = &cobra.Command{
	Use:   "start",
	Short: "Install the AppScope library",
	Long: `Install the AppScope library to:
/usr/lib/appscope/<version>/ with admin privileges, or 
/tmp/appscope/<version>/ otherwise`,
	Example: `scope start
scope start --rootdir /hostfs`,
	Args: cobra.NoArgs,
	Run: func(cmd *cobra.Command, args []string) {
		internal.InitConfig()
		rootdir, _ := cmd.Flags().GetString("rootdir")

		// Validate user has root permissions
		if rootdir != "" {
			if err := util.UserVerifyRootPerm(); err != nil {
				log.Error().Err(err)
				util.ErrAndExit("scope start with the --rootdir argument requires administrator privileges")
			}
		}

		if err := start.Start(rootdir); err != nil {
			util.ErrAndExit("Exiting due to start failure: %v", err)
		}
	},
}

func init() {
	startCmd.Flags().StringP("rootdir", "p", "", "Path to root filesystem of target namespace")
	RootCmd.AddCommand(startCmd)
}

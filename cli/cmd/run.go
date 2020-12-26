package cmd

import (
	"os"

	"github.com/criblio/scope/internal"
	"github.com/criblio/scope/run"
	"github.com/criblio/scope/util"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

// runCmd represents the run command
var runCmd = &cobra.Command{
	Use:     "run [flags]",
	Short:   "execute a scoped command",
	Long:    `Executes a scoped command`,
	Example: `scope run /bin/echo "foo"`,
	Args:    cobra.ArbitraryArgs,
	Run: func(cmd *cobra.Command, args []string) {
		run.Run(args)
	},
}

func init() {
	runCmd.Flags().BoolP("passthrough", "p", false, "Runs cscope with current environment & no config.")
	// This may be a bad assumption, if we have any args preceding this it might fail
	runCmd.SetFlagErrorFunc(func(cmd *cobra.Command, err error) error {
		internal.InitConfig(util.GetConfigPath())
		run.Run(os.Args[2:])
		return nil
	})
	viper.BindPFlag("passthrough", runCmd.Flags().Lookup("passthrough"))
	RootCmd.AddCommand(runCmd)
}

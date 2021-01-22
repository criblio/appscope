package cmd

import (
	"os"

	"github.com/criblio/scope/internal"
	"github.com/criblio/scope/run"
	"github.com/spf13/cobra"
)

// runCmd represents the run command
var runCmd = &cobra.Command{
	Use:     "run [flags]",
	Short:   "execute a scoped command",
	Long:    `Executes a scoped command`,
	Example: `scope run /bin/echo "foo"`,
	Args:    cobra.MinimumNArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		passthrough, _ := cmd.Flags().GetBool("passthrough")
		verbosity, _ := cmd.Flags().GetInt("verbosity")
		payloads, _ := cmd.Flags().GetBool("payloads")
		rc := run.Config{
			Passthrough: passthrough,
			Verbosity:   verbosity,
			Payloads:    payloads,
		}
		rc.Run(args)
	},
}

func init() {
	runCmd.Flags().Bool("passthrough", false, "Runs scopec with current environment & no config.")
	runCmd.Flags().IntP("verbosity", "v", 4, "Set scope metric verbosity")
	runCmd.Flags().BoolP("payloads", "p", false, "Capture payloads of network transactions")
	// This may be a bad assumption, if we have any args preceding this it might fail
	runCmd.SetFlagErrorFunc(func(cmd *cobra.Command, err error) error {
		internal.InitConfig()
		runCmd.Run(cmd, os.Args[2:])
		return nil
	})
	RootCmd.AddCommand(runCmd)
}

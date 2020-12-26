package cmd

import (
	"fmt"

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
		fmt.Printf("foo: %s\n", viper.GetString("foo"))
		fmt.Printf("%s\n", args)
	},
}

func init() {
	runCmd.Flags().BoolP("passthrough", "p", false, "Runs cscope with current environment & no config.")
	viper.BindPFlag("passthrough", runCmd.Flags().Lookup("passthrough"))
	RootCmd.AddCommand(runCmd)
}

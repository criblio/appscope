package cmd

import (
	"github.com/criblio/scope/bpf/sigdel"
	"github.com/spf13/cobra"
)

var daemonSummary bool
var daemonDate bool
var daemonTag bool

// daemonCmd represents the daemon command
var daemonCmd = &cobra.Command{
	Use:   "daemon [flags]",
	Short: "Handle system behavior",
	Long:  `Listem and respond to system events.`,
	Example: `scope daemon
scope daemon`,
	Args: cobra.NoArgs,
	Run: func(cmd *cobra.Command, args []string) {

		sigdel.Sigdel()
	},
}

func init() {
	RootCmd.AddCommand(daemonCmd)
}

package cmd

import (
	"fmt"
	"github.com/criblio/scope/bpf/sigdel"
	"github.com/criblio/scope/internal"
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
		summary := internal.GetGitSummary()
		date := internal.GetBuildDate()
		tag := internal.GetVersion()
		if daemonSummary {
			fmt.Printf("%s\n", summary)
			return
		}
		if daemonDate {
			fmt.Printf("%s\n", date)
			return
		}
		if daemonTag {
			fmt.Printf("%s\n", tag)
			return
		}
		fmt.Printf("Version: %s\n", summary)
		fmt.Printf("Build Date: %s\n", date)
	},


	Sigdel()
}

func init() {
	RootCmd.AddCommand(daemonCmd)
	daemonCmd.Flags().BoolVar(&daemonSummary, "summary", false, "Output just the summary")
	daemonCmd.Flags().BoolVar(&daemonDate, "date", false, "Output just the date")
	daemonCmd.Flags().BoolVar(&daemonTag, "tag", false, "Output just the tag")
}

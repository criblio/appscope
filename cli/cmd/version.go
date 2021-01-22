package cmd

import (
	"fmt"

	"github.com/criblio/scope/internal"
	"github.com/spf13/cobra"
)

var versionSummary bool
var versionDate bool

// versionCmd represents the version command
var versionCmd = &cobra.Command{
	Use:   "version [flags]",
	Short: "display scope version",
	Long:  `Outputs version info`,
	Example: `scope version
scope version --date
scope version --summary`,
	Args: cobra.NoArgs,
	Run: func(cmd *cobra.Command, args []string) {
		summary := internal.GetGitSummary()
		date := internal.GetBuildDate()
		if versionSummary {
			fmt.Printf("%s\n", summary)
			return
		}
		if versionDate {
			fmt.Printf("%s\n", date)
			return
		}
		fmt.Printf("Version: %s\n", summary)
		fmt.Printf("Build Date: %s\n", date)
	},
}

func init() {
	RootCmd.AddCommand(versionCmd)
	versionCmd.Flags().BoolVar(&versionSummary, "summary", false, "output just the summary")
	versionCmd.Flags().BoolVar(&versionDate, "date", false, "output just the date")
}

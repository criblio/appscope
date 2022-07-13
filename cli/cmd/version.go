package cmd

import (
	"fmt"

	"github.com/criblio/scope/internal"
	"github.com/spf13/cobra"
)

var versionSummary bool
var versionDate bool
var versionTag bool

// versionCmd represents the version command
var versionCmd = &cobra.Command{
	Use:   "version [flags]",
	Short: "Display scope version",
	Long:  `Outputs version info.`,
	Example: `scope version
scope version --date
scope version --summary
scope version --tag`,
	Args: cobra.NoArgs,
	Run: func(cmd *cobra.Command, args []string) {
		summary := internal.GetGitSummary()
		date := internal.GetBuildDate()
		tag := internal.GetVersion()
		if versionSummary {
			fmt.Printf("%s\n", summary)
			return
		}
		if versionDate {
			fmt.Printf("%s\n", date)
			return
		}
		if versionTag {
			fmt.Printf("%s\n", tag)
			return
		}
		fmt.Printf("Version: %s\n", summary)
		fmt.Printf("Build Date: %s\n", date)
	},
}

func init() {
	RootCmd.AddCommand(versionCmd)
	versionCmd.Flags().BoolVar(&versionSummary, "summary", false, "Output just the summary")
	versionCmd.Flags().BoolVar(&versionDate, "date", false, "Output just the date")
	versionCmd.Flags().BoolVar(&versionTag, "tag", false, "Output just the tag")
}

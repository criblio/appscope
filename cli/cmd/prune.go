package cmd

import (
	"fmt"
	"os"

	"github.com/criblio/scope/history"
	"github.com/criblio/scope/util"
	"github.com/spf13/cobra"
)

// pruneCmd represents the prune command
var pruneCmd = &cobra.Command{
	Use:   "prune [flags]",
	Short: "Prune deletes scope history",
	Long:  `Prune deletes scope history`,
	Example: `scope prune -k 20
scope prune -a`,
	Args: cobra.NoArgs,
	Run: func(cmd *cobra.Command, args []string) {
		keep, _ := cmd.Flags().GetInt("keep")
		all, _ := cmd.Flags().GetBool("all")
		force, _ := cmd.Flags().GetBool("force")

		count := ""
		if keep == -1 && all == false {
			util.ErrAndExit("Must specify keep or all")
		} else if all != false && keep > -1 {
			util.ErrAndExit("Cannot specify keep and all")
		} else if keep > -1 {
			count = fmt.Sprintf("all but %d", keep)
		} else if all != false {
			count = "all"
		}
		if !force {
			fmt.Printf("Are you sure you want to prune %s sessions? (y/yes): ", count)
			var response string
			_, err := fmt.Scanf("%s", &response)
			util.CheckErrSprintf(err, "error reading response: %v", err)
			if !(response == "y" || response == "yes") {
				os.Exit(0)
			}
		}
		sessions := history.GetSessions()
		countD := 0
		for idx, s := range sessions {
			if idx > len(sessions)-keep-1 {
				break
			}
			err := os.RemoveAll(s.WorkDir)
			util.CheckErrSprintf(err, "error removing work directory: %v", err)
			countD = idx + 1
		}
		fmt.Printf("Removed %d sessions.\n", countD)
		os.Exit(0)
	},
}

func init() {
	RootCmd.AddCommand(pruneCmd)
	pruneCmd.Flags().IntP("keep", "k", -1, "Keep last <keep> sessions")
	pruneCmd.Flags().BoolP("all", "a", false, "Delete all sessions")
	pruneCmd.Flags().BoolP("force", "f", false, "Do not prompt for confirmation")
}

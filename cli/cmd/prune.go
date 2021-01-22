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
		del, _ := cmd.Flags().GetInt("delete")

		count := ""
		if keep == -1 && del == -1 && all == false {
			util.ErrAndExit("Must specify keep, delete, or all")
		} else if all != false && keep > -1 {
			util.ErrAndExit("Cannot specify keep and all")
		} else if all != false && del > -1 {
			util.ErrAndExit("Cannot specify delete and all")
		} else if keep > -1 && del > -1 {
			util.ErrAndExit("Cannot specify delete and keep")
		} else if keep > -1 {
			count = fmt.Sprintf("all but %d", keep)
		} else if del > -1 {
			count = fmt.Sprintf("the last %d", del)
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
		if keep > -1 || all {
			for idx, s := range sessions {
				if idx > len(sessions)-keep-1 {
					break
				}
				err := os.RemoveAll(s.WorkDir)
				util.CheckErrSprintf(err, "error removing work directory: %v", err)
				countD = idx + 1
			}
		} else if del > -1 {
			sessions = sessions.Last(del)
			for idx, s := range sessions {
				err := os.RemoveAll(s.WorkDir)
				util.CheckErrSprintf(err, "error removing work directory: %v", err)
				countD = idx + 1
			}
		}
		fmt.Printf("Removed %d sessions.\n", countD)
		os.Exit(0)
	},
}

func init() {
	RootCmd.AddCommand(pruneCmd)
	pruneCmd.Flags().IntP("keep", "k", -1, "Keep last <keep> sessions")
	pruneCmd.Flags().IntP("delete", "d", -1, "Delete last <delete> sessions")
	pruneCmd.Flags().BoolP("all", "a", false, "Delete all sessions")
	pruneCmd.Flags().BoolP("force", "f", false, "Do not prompt for confirmation")
}

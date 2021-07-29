package cmd

import (
	"bufio"
	"fmt"
	"os"

	"github.com/criblio/scope/history"
	"github.com/spf13/cobra"
)

// pruneCmd represents the prune command
var pruneCmd = &cobra.Command{
	Use:   "prune [flags]",
	Short: "Prune deletes session history",
	Long:  `Scope stores captured data in a directory per session. Prune allows you to delete prior sessions. `,
	Example: `scope prune -k 20
scope prune -a`,
	Args: cobra.NoArgs,
	Run: func(cmd *cobra.Command, args []string) {
		keep, _ := cmd.Flags().GetInt("keep")
		all, _ := cmd.Flags().GetBool("all")
		force, _ := cmd.Flags().GetBool("force")
		del, _ := cmd.Flags().GetInt("delete")

		count := ""

		if keep < 0 || del < 0 {
			helpErrAndExit(cmd, "Must be a positive number")
		} else if keep == 0 && del == 0 && !all {
			helpErrAndExit(cmd, "Must specify keep, delete, or all")
		} else if all && keep > 0 {
			helpErrAndExit(cmd, "Cannot specify keep and all")
		} else if all && del > 0 {
			helpErrAndExit(cmd, "Cannot specify delete and all")
		} else if keep > 0 && del > 0 {
			helpErrAndExit(cmd, "Cannot specify delete and keep")
		} else if keep > 0 {
			count = fmt.Sprintf("all but %d", keep)
		} else if del > 0 {
			count = fmt.Sprintf("the last %d", del)
		} else if all {
			count = "all"
		}
		if !force {
			fmt.Printf("Are you sure you want to prune %s sessions? (y/yes): ", count)
			in := bufio.NewScanner(os.Stdin)
			in.Scan()
			response := in.Text()
			if !(response == "y" || response == "yes") {
				fmt.Printf("Removed 0 sessions.\n")
				os.Exit(0)
			}
		}
		sessions := history.GetSessions()
		countD := 0
		if all {
			countD = len(sessions)
			sessions.Remove()
		} else if keep > 0 {
			toRemove := len(sessions) - keep
			removeS := sessions.First(toRemove)
			countD = len(removeS)
			removeS.Remove()
		} else if del > 0 {
			removeS := sessions.Last(del)
			countD = len(removeS)
			removeS.Remove()
		}
		fmt.Printf("Removed %d sessions.\n", countD)
		os.Exit(0)
	},
}

func init() {
	RootCmd.AddCommand(pruneCmd)
	pruneCmd.Flags().IntP("keep", "k", 0, "Keep last <keep> sessions")
	pruneCmd.Flags().IntP("delete", "d", 0, "Delete last <delete> sessions")
	pruneCmd.Flags().BoolP("all", "a", false, "Delete all sessions")
	pruneCmd.Flags().BoolP("force", "f", false, "Do not prompt for confirmation")
}

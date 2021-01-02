package cmd

import (
	"fmt"
	"os"
	"time"

	"github.com/criblio/scope/history"
	"github.com/criblio/scope/util"
	"github.com/kballard/go-shellquote"
	"github.com/spf13/cobra"
)

// historyCmd represents the history command
var historyCmd = &cobra.Command{
	Use:     "history [flags]",
	Short:   "List scope session history",
	Long:    `List scope session history`,
	Example: `scope history`,
	Aliases: []string{"hist"},
	Args:    cobra.NoArgs,
	Run: func(cmd *cobra.Command, args []string) {
		all, _ := cmd.Flags().GetBool("all")
		last, _ := cmd.Flags().GetInt("last")
		running, _ := cmd.Flags().GetBool("running")
		id, _ := cmd.Flags().GetInt("id")
		onlydir, _ := cmd.Flags().GetBool("dir")
		sessions := history.GetSessions()
		if all {
			fmt.Println("Displaying all sessions")
		} else {
			if !onlydir && !running {
				fmt.Printf("Displaying last %d sessions\n", last)
			}
			sessions = sessions.Last(last)
		}
		if running {
			fmt.Print("Displaying running sessions\n")
			sessions = sessions.Running()
		}
		sessions = sessions.Args()
		if id > -1 {
			s := sessions.ID(id)
			if len(s) == 0 {
				util.ErrAndExit("session id %d not found", id)
			}
			session := s[0]
			if onlydir {
				fmt.Printf("%s\n", session.WorkDir)
				os.Exit(0)
			}
			util.PrintObj([]util.ObjField{
				{Name: "ID", Field: "id"},
				{Name: "Command", Field: "cmd"},
				{Name: "Cmdline", Field: "args", Transform: func(obj interface{}) string { return util.TruncWithElipsis(shellquote.Join(obj.([]string)...), 25) }},
				{Name: "PID", Field: "pid"},
				{Name: "Timestamp", Field: "timestamp", Transform: func(obj interface{}) string { return time.Unix(int64(0), obj.(int64)).Format(time.Stamp) }},
				{Name: "WorkDir", Field: "workDir"},
				{Name: "ArgsPath", Field: "argspath"},
				{Name: "CmdDirPath", Field: "cmddirpath"},
				{Name: "EventsPath", Field: "eventspath"},
				{Name: "MetricsPath", Field: "metricspath"},
			}, session)
		} else {
			util.PrintObj([]util.ObjField{
				{Name: "ID", Field: "id"},
				{Name: "Command", Field: "cmd"},
				{Name: "Cmdline", Field: "args", Transform: func(obj interface{}) string { return util.TruncWithElipsis(shellquote.Join(obj.([]string)...), 25) }},
				{Name: "PID", Field: "pid"},
				{Name: "Age", Field: "timestamp", Transform: func(obj interface{}) string { return util.GetAgo(time.Unix(0, obj.(int64))) }},
			}, sessions)
		}
	},
}

func init() {
	historyCmd.Flags().BoolP("all", "a", false, "List all sessions")
	historyCmd.Flags().BoolP("running", "r", false, "List running sessions")
	historyCmd.Flags().IntP("last", "n", 20, "Show last <n> sessions")
	historyCmd.Flags().IntP("id", "i", -1, "Display info from specific from session ID")
	historyCmd.Flags().BoolP("dir", "d", false, "Output just directory (with -i)")
	RootCmd.AddCommand(historyCmd)
}

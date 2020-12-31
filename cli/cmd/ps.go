package cmd

import (
	"fmt"
	"os"
	"time"

	"github.com/criblio/scope/ps"
	"github.com/criblio/scope/util"
	"github.com/kballard/go-shellquote"
	"github.com/spf13/cobra"
)

// psCmd represents the ps command
var psCmd = &cobra.Command{
	Use:     "ps [flags]",
	Short:   "List scope sessions",
	Long:    `List scope sessions`,
	Example: `scope ps`,
	Args:    cobra.NoArgs,
	Run: func(cmd *cobra.Command, args []string) {
		all, _ := cmd.Flags().GetBool("all")
		last, _ := cmd.Flags().GetInt("last")
		id, _ := cmd.Flags().GetInt("id")
		onlydir, _ := cmd.Flags().GetBool("dir")
		sessions := ps.GetSessions().Last(last)
		if !all {
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
	psCmd.Flags().BoolP("all", "a", false, "List all sessions, including non-running sessions")
	psCmd.Flags().IntP("last", "n", -1, "Show n last sessions")
	psCmd.Flags().Int("id", -1, "Display info from specific from session ID")
	psCmd.Flags().BoolP("dir", "d", false, "Output just directory (with --id)")
	RootCmd.AddCommand(psCmd)
}

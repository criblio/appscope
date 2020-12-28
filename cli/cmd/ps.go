package cmd

import (
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
		sessions := ps.GetSessions().Last(last)
		if !all {
			sessions = sessions.Running()
		}
		sessions = sessions.Args()
		util.PrintObj([]util.ObjField{
			{Name: "ID", Field: "id"},
			{Name: "Command", Field: "cmd"},
			{Name: "Cmdline", Field: "args", Transform: func(obj interface{}) string { return util.TruncWithElipsis(shellquote.Join(obj.([]string)...), 25) }},
			{Name: "PID", Field: "pid"},
			{Name: "Age", Field: "timestamp", Transform: func(obj interface{}) string { return util.GetAgo(time.Unix(0, obj.(int64))) }},
		}, sessions)
	},
}

func init() {
	psCmd.Flags().BoolP("all", "a", false, "List all sessions, including non-running sessions")
	psCmd.Flags().IntP("last", "n", -1, "Show n last sessions")
	RootCmd.AddCommand(psCmd)
}

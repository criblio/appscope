package cmd

import (
	"time"

	"github.com/criblio/scope/internal"
	"github.com/criblio/scope/util"
	"github.com/spf13/cobra"
)

// watchCmd represents the watch command
var watchCmd = &cobra.Command{
	Use:   "watch [flags]",
	Short: "Executes a scoped command on an interval",
	Long: `Watch executes a scoped command on an interval. Note, when calling the watch subcommand you should call it 
like scope watch -- <command>, to avoid scope attempting to parse flags passed to the executed command.`,
	Example: `scope watch -i 5s -- /bin/echo "foo"
scope watch --interval=1m-- perl -e 'print "foo\n"'
scope watch --interval=5s --payloads -- nc -lp 10001
scope watch -i 1h -- curl https://wttr.in/94105
scope watch --interval=10s -- curl https://wttr.in/94105`,
	Args: cobra.MinimumNArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		interval, _ := cmd.Flags().GetString("interval")
		internal.InitConfig()
		rc.Subprocess = true
		dur, err := time.ParseDuration(interval)
		util.CheckErrSprintf(err, "error parsing time duration string \"%s\": %v", interval, err)
		timer := time.Tick(dur)
		rc.Run(args, false)
		for range timer {
			rc.Run(args, false)
		}
	},
}

func init() {
	runCmdFlags(watchCmd, rc)
	watchCmd.Flags().StringP("interval", "i", "", "Run every <x>(s|m|h)")
	watchCmd.MarkFlagRequired("interval")
	RootCmd.AddCommand(watchCmd)
}

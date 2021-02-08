package cmd

import (
	"os"

	"github.com/criblio/scope/internal"
	"github.com/criblio/scope/run"
	"github.com/spf13/cobra"
)

var rc *run.Config = &run.Config{}

// runCmd represents the run command
var runCmd = &cobra.Command{
	Use:   "run [flags]",
	Short: "Executes a scoped command",
	Long: `Run executes a scoped command. By default, calling scope with no subcommands will execute run for args after 
scope. However, scope allows for additional arguments to be passed to run to capture payloads or to up metric 
verbosity. Note, when calling the run subcommand you should call it like scope run -- <command>, to avoid scope 
attempting to parse flags passed to the executed command.`,
	Example: `scope run -- /bin/echo "foo"
scope run -- perl -e 'print "foo\n"'
scope run --payloads -- nc -lp 10001
scope run -- curl https://wttr.in/94105`,
	Args: cobra.MinimumNArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		rc.Run(args)
	},
}

func init() {
	runCmd.Flags().BoolVar(&rc.Passthrough, "passthrough", false, "Runs scopec with current environment & no config.")
	runCmd.Flags().IntVarP(&rc.Verbosity, "verbosity", "v", 4, "Set scope metric verbosity")
	runCmd.Flags().BoolVarP(&rc.Payloads, "payloads", "p", false, "Capture payloads of network transactions")
	runCmd.Flags().StringVar(&rc.MetricsFormat, "metricformat", "ndjson", "Set format of metrics output (statsd|ndjson)")
	runCmd.Flags().StringVarP(&rc.MetricsDest, "metricdest", "m", "", "Set destination for metrics (tcp://host:port, udp://host:port, or file:///path/file.json)")
	runCmd.Flags().StringVarP(&rc.EventsDest, "eventdest", "e", "", "Set destination for events (tcp://host:port, udp://host:port, or file:///path/file.json)")
	// This may be a bad assumption, if we have any args preceding this it might fail
	runCmd.SetFlagErrorFunc(func(cmd *cobra.Command, err error) error {
		internal.InitConfig()
		runCmd.Run(cmd, os.Args[2:])
		return nil
	})
	RootCmd.AddCommand(runCmd)
}

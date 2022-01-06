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
	Use:   "run [flags] [command]",
	Short: "Executes a scoped command",
	Long: `Run executes a scoped command. By default, calling scope with no subcommands will execute run for args after 
scope. However, scope allows for additional arguments to be passed to run to capture payloads or to up metric 
verbosity. Note, when calling the run subcommand you should call it like scope run -- <command>, to avoid scope 
attempting to parse flags passed to the executed command.

The --*dest flags accept file names like /tmp/scope.log or URLs like file:///tmp/scope.log. They may also
be set to sockets with unix:///var/run/mysock, tcp://hostname:port, udp://hostname:port, or tls://hostname:port.`,
	Example: `scope run -- /bin/echo "foo"
scope run -- perl -e 'print "foo\n"'
scope run --payloads -- nc -lp 10001
scope run -- curl https://wttr.in/94105
scope run -c tcp://127.0.0.1:10091 -- curl https://wttr.in/94105
scope run -c edge -- top`,
	Args: cobra.MinimumNArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		internal.InitConfig()
		rc.Run(args)
	},
}

func init() {
	runCmdFlags(runCmd, rc)
	// This may be a bad assumption, if we have any args preceding this it might fail
	runCmd.SetFlagErrorFunc(func(cmd *cobra.Command, err error) error {
		internal.InitConfig()
		runCmd.Run(cmd, os.Args[2:])
		return nil
	})
	RootCmd.AddCommand(runCmd)
}

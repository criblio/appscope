package cmd

import (
	"os"

	"github.com/criblio/scope/internal"
	"github.com/criblio/scope/run"
	"github.com/spf13/cobra"
)

/* Args Matrix (X disallows)
 *                 cribldest metricformat metricdest eventdest nobreaker authtoken verbosity payloads loglevel librarypath userconfig
 * cribldest       -                      X          X                                                                     X
 * metricformat              -                                                                                             X
 * metricdest      X                      -                                                                                X
 * eventdest       X                                 -                                                                     X
 * nobreaker                                                   -                                                           X
 * authtoken                                                             -                                                 X
 * verbosity                                                                       -                                       X
 * payloads                                                                                  -                             X
 * loglevel                                                                                           -                    X
 * librarypath                                                                                                 -
 * userconfig      X         X            X          X         X         X         X         X        X                    -
 */

var rc *run.Config = &run.Config{}

// runCmd represents the run command
var runCmd = &cobra.Command{
	Use:   "run [flags] [command]",
	Short: "Executes a scoped command",
	Long: `Executes a scoped command. By default, calling scope with no subcommands will run the executables you pass as arguments to scope. However, scope allows for additional arguments to be passed to run to capture payloads or to increase metrics' verbosity. Must be called with the -- flag, e.g., 'scope run -- <command>', to prevent AppScope from attempting to parse flags passed to the executed command.

The --*dest flags accept file names like /tmp/scope.log; URLs like file:///tmp/scope.log; or sockets specified with the pattern unix:///var/run/mysock, tcp://hostname:port, udp://hostname:port, or tls://hostname:port.`,
	Example: `scope run -- /bin/echo "foo"
scope run -- perl -e 'print "foo\n"'
scope run --payloads -- nc -lp 10001
scope run -- curl https://wttr.in/94105
scope run -c tcp://127.0.0.1:10091 -- curl https://wttr.in/94105
scope run -c edge -- top`,
	Args: cobra.MinimumNArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		internal.InitConfig()

		// Disallow bad argument combinations (see Arg Matrix at top of file)
		if rc.CriblDest != "" && rc.MetricsDest != "" {
			helpErrAndExit(cmd, "Cannot specify --cribldest and --metricdest")
		} else if rc.CriblDest != "" && rc.EventsDest != "" {
			helpErrAndExit(cmd, "Cannot specify --cribldest and --eventdest")
		} else if rc.CriblDest != "" && rc.UserConfig != "" {
			helpErrAndExit(cmd, "Cannot specify --cribldest and --userconfig")
		} else if cmd.Flags().Lookup("metricformat").Changed && rc.UserConfig != "" {
			helpErrAndExit(cmd, "Cannot specify --metricformat and --userconfig")
		} else if rc.EventsDest != "" && rc.UserConfig != "" {
			helpErrAndExit(cmd, "Cannot specify --eventdest and --userconfig")
		} else if rc.NoBreaker && rc.UserConfig != "" {
			helpErrAndExit(cmd, "Cannot specify --nobreaker and --userconfig")
		} else if rc.AuthToken != "" && rc.UserConfig != "" {
			helpErrAndExit(cmd, "Cannot specify --authtoken and --userconfig")
		} else if cmd.Flags().Lookup("verbosity").Changed && rc.UserConfig != "" {
			helpErrAndExit(cmd, "Cannot specify --verbosity and --userconfig")
		} else if rc.Payloads && rc.UserConfig != "" {
			helpErrAndExit(cmd, "Cannot specify --payloads and --userconfig")
		} else if rc.Loglevel != "" && rc.UserConfig != "" {
			helpErrAndExit(cmd, "Cannot specify --loglevel and --userconfig")
		}

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

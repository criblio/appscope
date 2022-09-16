package cmd

import (
	"github.com/criblio/scope/internal"
	"github.com/spf13/cobra"
)

/* Args Matrix (X disallows)
 *                 cribldest metricformat metricdest eventdest nobreaker authtoken passthrough verbosity payloads loglevel librarypath userconfig
 * cribldest       -                      X          X                             X                                                   X
 * metricformat              -                                                     X                                                   X
 * metricdest      X                      -                                        X                                                   X
 * eventdest       X                                 -                             X                                                   X
 * nobreaker                                                   -                   X                                                   X
 * authtoken                                                             -         X                                                   X
 * passthrough     X         X            X          X         X         X         -           X         X        X                    X
 * verbosity                                                                       X           -                                       X
 * payloads                                                                        X                     -                             X
 * loglevel                                                                        X                              -                    X
 * librarypath                                                                                                             -
 * userconfig      X         X            X          X         X         X         X           X         X        X                    -
 */

// attachCmd represents the run command
var attachCmd = &cobra.Command{
	Use:   "attach [flags] PID | <process_name>",
	Short: "Scope a currently-running process",
	Long: `Scopes a currently-running process identified by PID or <process_name>.

The --*dest flags accept file names like /tmp/scope.log or URLs like file:///tmp/scope.log. They may also
be set to sockets with unix:///var/run/mysock, tcp://hostname:port, udp://hostname:port, or tls://hostname:port.`,
	Example: `scope attach 1000
scope attach firefox 
scope attach --payloads 2000`,
	Args: cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		internal.InitConfig()

		// Disallow bad argument combinations (see Arg Matrix at top of file)
		if rc.CriblDest != "" && rc.MetricsDest != "" {
			helpErrAndExit(cmd, "Cannot specify --cribldest and --metricdest")
		} else if rc.CriblDest != "" && rc.EventsDest != "" {
			helpErrAndExit(cmd, "Cannot specify --cribldest and --eventdest")
		} else if rc.CriblDest != "" && rc.Passthrough {
			helpErrAndExit(cmd, "Cannot specify --cribldest and --passthrough")
		} else if rc.CriblDest != "" && rc.UserConfig != "" {
			helpErrAndExit(cmd, "Cannot specify --cribldest and --userconfig")
		} else if cmd.Flags().Lookup("metricformat").Changed && rc.Passthrough {
			helpErrAndExit(cmd, "Cannot specify --metricformat and --passthrough")
		} else if cmd.Flags().Lookup("metricformat").Changed && rc.UserConfig != "" {
			helpErrAndExit(cmd, "Cannot specify --metricformat and --userconfig")
		} else if rc.EventsDest != "" && rc.Passthrough {
			helpErrAndExit(cmd, "Cannot specify --eventdest and --passthrough")
		} else if rc.EventsDest != "" && rc.UserConfig != "" {
			helpErrAndExit(cmd, "Cannot specify --eventdest and --userconfig")
		} else if rc.NoBreaker && rc.Passthrough {
			helpErrAndExit(cmd, "Cannot specify --nobreaker and --passthrough")
		} else if rc.NoBreaker && rc.UserConfig != "" {
			helpErrAndExit(cmd, "Cannot specify --nobreaker and --userconfig")
		} else if rc.AuthToken != "" && rc.Passthrough {
			helpErrAndExit(cmd, "Cannot specify --authtoken and --passthrough")
		} else if rc.AuthToken != "" && rc.UserConfig != "" {
			helpErrAndExit(cmd, "Cannot specify --authtoken and --userconfig")
		} else if rc.Passthrough && cmd.Flags().Lookup("verbosity").Changed {
			helpErrAndExit(cmd, "Cannot specify --passthrough and --verbosity")
		} else if rc.Passthrough && rc.Payloads {
			helpErrAndExit(cmd, "Cannot specify --passthrough and --payloads")
		} else if rc.Passthrough && rc.Loglevel != "" {
			helpErrAndExit(cmd, "Cannot specify --passthrough and --loglevel")
		} else if rc.Passthrough && rc.UserConfig != "" {
			helpErrAndExit(cmd, "Cannot specify --passthrough and --userconfig")
		} else if cmd.Flags().Lookup("verbosity").Changed && rc.UserConfig != "" {
			helpErrAndExit(cmd, "Cannot specify --verbosity and --userconfig")
		} else if rc.Payloads && rc.UserConfig != "" {
			helpErrAndExit(cmd, "Cannot specify --payloads and --userconfig")
		} else if rc.Loglevel != "" && rc.UserConfig != "" {
			helpErrAndExit(cmd, "Cannot specify --loglevel and --userconfig")
		}

		return rc.Attach(args)
	},
}

func init() {
	runCmdFlags(attachCmd, rc)
	RootCmd.AddCommand(attachCmd)
}

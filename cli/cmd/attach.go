package cmd

import (
	"github.com/criblio/scope/internal"
	"github.com/spf13/cobra"
)

// attachCmd represents the run command
var attachCmd = &cobra.Command{
	Use:   "attach [flags] [PID/Process]",
	Short: "Scope an existing process",
	Long: `Attach scopes an existing process. Scope allows for additional arguments to be passed to attach to capture payloads or to up metric 
verbosity. 

The --*dest flags accept file names like /tmp/scope.log or URLs like file:///tmp/scope.log. They may also
be set to sockets with unix:///var/run/mysock, tcp://hostname:port, udp://hostname:port, or tls://hostname:port.`,
	Example: `scope attach 1000
scope attach firefox 
scope attach --payloads 2000`,
	Args: cobra.ExactArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		internal.InitConfig()
		rc.Attach(args)
	},
}

func init() {
	runCmdFlags(attachCmd, rc)
	RootCmd.AddCommand(attachCmd)
}

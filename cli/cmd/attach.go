package cmd

import (
	"os"

	"github.com/criblio/scope/internal"
	"github.com/spf13/cobra"
)

// attachCmd represents the run command
var attachCmd = &cobra.Command{
	Use:   "attach [flags] [PID]",
	Short: "Scope an existing PID",
	Long: `Attach scopes an existing PID. Scope allows for additional arguments to be passed to attach to capture payloads or to up metric 
verbosity. 

The --*dest flags accept file names like /tmp/scope.log or URLs like file:///tmp/scope.log. They may also
be set to sockets with tcp://hostname:port, udp://hostname:port, or tls://hostname:port.`,
	Example: `scope attach 1000
scope attach --payloads 2000`,
	Args: cobra.ExactArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		internal.InitConfig()
		rc.Run(args, true)
	},
}

func init() {
	runCmdFlags(attachCmd, rc)
	// This may be a bad assumption, if we have any args preceding this it might fail
	attachCmd.SetFlagErrorFunc(func(cmd *cobra.Command, err error) error {
		internal.InitConfig()
		attachCmd.Run(cmd, os.Args[2:])
		return nil
	})
	RootCmd.AddCommand(attachCmd)
}

package cmd

import (
	"fmt"

	"github.com/criblio/scope/bpf/sigdel"
	"github.com/spf13/cobra"
)

// daemonCmd represents the daemon command
var daemonCmd = &cobra.Command{
	Use:     "daemon [flags]",
	Short:   "Handle system behavior",
	Long:    `Listen and respond to system events.`,
	Example: `scope daemon`,
	Args:    cobra.NoArgs,
	Run: func(cmd *cobra.Command, args []string) {
		sigEventChan := make(chan sigdel.SigEvent)
		go sigdel.Sigdel(sigEventChan)

		for {
			select {
			case sigEvent := <-sigEventChan:
				fmt.Printf("Signal CPU: %02d signal %d errno %d handler 0x%x pid: %d uid: %d gid: %d app %s\n",
					sigEvent.CPU, sigEvent.Sig, sigEvent.Errno, sigEvent.Handler,
					sigEvent.Pid, sigEvent.Uid, sigEvent.Gid, sigEvent.Comm)

				/*
					if err := crash.GenFiles(sigEvent.Pid, sigEvent.Sig, sigEvent.Errno); err != nil {
						log.Error().Err(err)
						util.ErrAndExit("error generating crash files")
					}
				*/

			}
		}
	},
}

func init() {
	RootCmd.AddCommand(daemonCmd)
}

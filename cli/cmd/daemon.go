package cmd

import (
	"fmt"

	"github.com/criblio/scope/daemon"
	"github.com/criblio/scope/util"
	"github.com/rs/zerolog/log"
	"github.com/spf13/cobra"
)

/* Args Matrix (X disallows)
 *                 filedest 	sendcore
 * filedest        -
 * sendcore                     -
 */

// daemonCmd represents the daemon command
var daemonCmd = &cobra.Command{
	Use:     "daemon [flags]",
	Short:   "Run the scope daemon",
	Long:    `Listen and respond to system events.`,
	Example: `scope daemon`,
	Args:    cobra.NoArgs,
	Run: func(cmd *cobra.Command, args []string) {
		filedest, _ := cmd.Flags().GetString("filedest")
		sendcore, _ := cmd.Flags().GetBool("sendcore")

		d := daemon.New(filedest)

		sigEventChan := make(chan int, 2)
		sigEventChan <- 150811
		// sigEventChan := make(chan sigdel.SigEvent)
		// go sigdel.Sigdel(sigEventChan)

		for {
			select {
			/*
				case sigEvent := <-sigEventChan:
					// Signal received
					log.Info().Msgf("Signal CPU: %02d signal %d errno %d handler 0x%x pid: %d uid: %d gid: %d app %s\n",
					sigEvent.CPU, sigEvent.Sig, sigEvent.Errno, sigEvent.Handler,
					sigEvent.Pid, sigEvent.Uid, sigEvent.Gid, sigEvent.Comm)

					// Generate/retrieve crash files
					if err := crash.GenFiles(sigEvent.Pid, sigEvent.Sig, sigEvent.Errno); err != nil {
						log.Error().Err(err)
						util.ErrAndExit("error generating crash files")
					}

					// If a network destination is specified, send crash files
					// See below example usage
			*/
			case pid := <-sigEventChan:
				if filedest != "" {
					if err := d.Connect(); err != nil {
						log.Error().Err(err)
						util.ErrAndExit("error starting daemon")
					}

					files := []string{
						fmt.Sprintf("/tmp/appscope/%d/snapshot", pid),
						fmt.Sprintf("/tmp/appscope/%d/backtrace", pid),
						fmt.Sprintf("/tmp/appscope/%d/info", pid),
						fmt.Sprintf("/tmp/appscope/%d/cfg", pid),
					}
					if sendcore {
						files = append(files, fmt.Sprintf("/tmp/appscope/%d/core", pid))
					}

					if err := d.SendFiles(files); err != nil {
						log.Error().Err(err)
						util.ErrAndExit("error sending files to %s", filedest)
					}

					d.Disconnect()
				}
			}
		}
	},
}

func init() {
	daemonCmd.Flags().StringP("filedest", "f", "", "Set destination for files (host:port defaults to tcp://)")
	daemonCmd.Flags().BoolP("sendcore", "s", false, "Include core file when sending files to network destination")
	RootCmd.AddCommand(daemonCmd)
}

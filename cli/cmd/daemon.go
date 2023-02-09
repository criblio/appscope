package cmd

import (
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

		/*
			sigEventChan := make(chan sigdel.SigEvent)
			go sigdel.Sigdel(sigEventChan)

			for {
				select {
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
				}
			}
		*/

		// Example usage
		if filedest != "" {
			if err := d.Connect(); err != nil {
				log.Error().Err(err)
				util.ErrAndExit("error starting daemon")
			}
			defer d.Disconnect()

			files := []string{
				"/tmp/appscope/150811/snapshot",
				"/tmp/appscope/150811/backtrace",
				"/tmp/appscope/150811/info",
				"/tmp/appscope/150811/cfg",
			}
			if sendcore {
				files = append(files, "/tmp/appscope/150811/core")
			}

			if err := d.SendFiles(files); err != nil {
				log.Error().Err(err)
				util.ErrAndExit("error sending files to %s", filedest)
			}
		}

	},
}

func init() {
	daemonCmd.Flags().StringP("filedest", "f", "", "Set destination for files (host:port defaults to tls://)")
	daemonCmd.Flags().BoolP("sendcore", "s", false, "Include core file when sending files to network destination")
	RootCmd.AddCommand(daemonCmd)
}

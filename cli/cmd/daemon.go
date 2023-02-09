package cmd

import (
	"github.com/spf13/cobra"
)

/* Args Matrix (X disallows)
 *                 filedest
 * filedest        -
 */

// daemonCmd represents the daemon command
var daemonCmd = &cobra.Command{
	Use:     "daemon [flags]",
	Short:   "Handle system behavior",
	Long:    `Listen and respond to system events.`,
	Example: `scope daemon`,
	Args:    cobra.NoArgs,
	Run: func(cmd *cobra.Command, args []string) {
		filedest, _ := cmd.Flags().GetString("filedest")

		// Example usage
		if filedest != "" {
			if err := daemon.SendFiles("/tmp/appscope/1", filedest); err != nil {
				return err
			}
		}

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
					if filedest != "" {
						if err := daemon.SendFiles("/tmp/appscope/1", filedest); err != nil {
							return err
						}
					}
				}
			}
		*/
	},
}

func init() {
	daemonCmd.Flags().StringP("filedest", "f", "", "Set destination for files (host:port defaults to tls://)")
	RootCmd.AddCommand(daemonCmd)
}

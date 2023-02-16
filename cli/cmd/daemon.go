package cmd

import (
	"fmt"
	"os"

	"github.com/criblio/scope/bpf/sigdel"
	"github.com/criblio/scope/daemon"
	"github.com/criblio/scope/ipc"
	"github.com/criblio/scope/snapshot"
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

		// Create a history directory for logs
		snapshot.CreateWorkDir("daemon")

		// Validate user has root permissions
		if err := util.UserVerifyRootPerm(); err != nil {
			log.Error().Err(err)
			util.ErrAndExit("scope daemon requires administrator privileges")
		}

		// Buffered Channel (non-blocking until full)
		sigEventChan := make(chan sigdel.SigEvent, 128)
		go sigdel.Sigdel(sigEventChan)

		d := daemon.New(filedest)
		for {
			select {
			case sigEvent := <-sigEventChan:
				// Signal received
				log.Info().Msgf("Signal CPU: %02d signal %d errno %d handler 0x%x pid: %d nspid: %d uid: %d gid: %d app %s\n",
					sigEvent.CPU, sigEvent.Sig, sigEvent.Errno, sigEvent.Handler,
					sigEvent.Pid, sigEvent.NsPid, sigEvent.Uid, sigEvent.Gid, sigEvent.Comm)

				// TODO Filter out expected signals from libscope
				// get proc/pid/maps from ns
				// iterate through, find libscope and get range
				// is signal handler address in libscopeStart-libscopeEnd range

				// If the daemon is running on the host, use the pid(hostpid) to get the crash files
				// If the daemon is running in a container, use the nspid to get the crash files
				daemonInNs, _, err := ipc.IpcNsLastPidFromPid(ipc.IpcPidCtx{Pid: os.Getpid()})
				if err != nil {
					log.Error().Err(err)
					util.ErrAndExit("error determining if daemon is in container")
				}
				var pid uint32
				if daemonInNs {
					pid = sigEvent.NsPid
				} else {
					pid = sigEvent.Pid
				}

				// Generate/retrieve snapshot files
				if err := snapshot.GenFiles(sigEvent.Sig, sigEvent.Errno, pid, sigEvent.Uid, sigEvent.Gid, sigEvent.Handler, string(sigEvent.Comm[:]), ""); err != nil {
					log.Error().Err(err)
					util.ErrAndExit("error generating crash files")
				}

				// If a network destination is specified, send crash files
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
					if err := d.SendFiles(files, true); err != nil {
						log.Error().Err(err)
						util.ErrAndExit("error sending files to %s", filedest)
					}

					if sendcore {
						files = []string{
							fmt.Sprintf("/tmp/appscope/%d/core", pid),
						}
						if err := d.SendFiles(files, false); err != nil {
							log.Error().Err(err)
							util.ErrAndExit("error sending core file to %s", filedest)
						}
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

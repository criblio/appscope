package cmd

import (
	"errors"
	"strconv"

	"github.com/criblio/scope/crash"
	"github.com/criblio/scope/internal"
	"github.com/criblio/scope/util"
	"github.com/rs/zerolog/log"
	"github.com/shirou/gopsutil/v3/process"
	"github.com/spf13/cobra"
)

/* Args Matrix (X disallows)
 */

// snapshotCmd represents the ps command
var snapshotCmd = &cobra.Command{
	Use:   "snapshot [PID]",
	Short: "Create a snapshot for a process",
	Long:  `Create a snapshot for a process. Snapshot file/s will be created in /tmp/appscope/[PID]/.`,
	Args:  cobra.ExactArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		internal.InitConfig()
		// Nice message for non-adminstrators
		err := util.UserVerifyRootPerm()
		if errors.Is(err, util.ErrGetCurrentUser) {
			util.ErrAndExit("Unable to get current user: %v", err)
		}
		if errors.Is(err, util.ErrMissingAdmPriv) {
			util.ErrAndExit(err.Error())
		}

		pid, err := strconv.Atoi(args[0])
		if err != nil {
			util.ErrAndExit("error parsing PID argument")
		}

		// Get process properties
		p, err := process.NewProcess(int32(pid))
		if err != nil {
			log.Error().Err(err).Msgf("error getting process for pid %d", pid)
			util.ErrAndExit(err.Error())
		}
		uids, err := p.Uids()
		if err != nil {
			log.Error().Err(err).Msgf("error getting uid for pid %d", pid)
			util.ErrAndExit(err.Error())
		}
		gids, err := p.Gids()
		if err != nil {
			log.Error().Err(err).Msgf("error getting gid for pid %d", pid)
			util.ErrAndExit(err.Error())
		}
		procName, err := p.Name()
		if err != nil {
			log.Error().Err(err).Msgf("error getting name for pid %d", pid)
			util.ErrAndExit(err.Error())
		}
		procArgs, err := p.Cmdline()
		if err != nil {
			log.Error().Err(err).Msgf("error getting args for pid %d", pid)
			util.ErrAndExit(err.Error())
		}

		err = crash.GenFiles(0, 0, uint32(pid), uint32(uids[0]), uint32(gids[0]), "", procName, procArgs)
		if err != nil {
			util.ErrAndExit(err.Error())
		}
	},
}

func init() {
	RootCmd.AddCommand(snapshotCmd)
}

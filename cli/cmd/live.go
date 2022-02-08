package cmd

import (
	"context"
	"os"
	"time"

	"github.com/criblio/scope/internal"
	"github.com/criblio/scope/live"
	"github.com/criblio/scope/run"
	"github.com/criblio/scope/util"

	//	"github.com/criblio/scope/web"
	"github.com/spf13/cobra"
	"golang.org/x/sync/errgroup"
)

// liveCmd represents the live command
var liveCmd = &cobra.Command{
	Use:     "live",
	Short:   "View a scope session live in your browser",
	Long:    `View a scope session live in your browser.`,
	Example: `scope live`,
	Args:    cobra.MinimumNArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		internal.InitConfig()

		s := live.NewScope()
		ctx := context.Context(context.Background())
		g, gctx := errgroup.WithContext(ctx)

		g.Go(util.Signal(gctx))
		g.Go(live.Receiver(gctx, g, s))
		//		g.Go(web.Server(gctx, g, s))

		// Wait until we're listening before run is called
		// TODO channel to unblock when listener is ready
		time.Sleep(2 * time.Second)

		lrc := &run.Config{
			MetricsFormat: "ndjson",
			MetricsDest:   "unix://@scopelive",
			EventsDest:    "unix://@scopelive",
			Verbosity:     4,
			Subprocess:    true,
		}
		g.Go(lrc.LiveRun(gctx, g, args))

		if err := g.Wait(); err != nil {
			util.ErrAndExit("Exit reason: " + err.Error())
		}
	},
}

func init() {
	// runCmdFlags(liveCmd, rc)

	// This may be a bad assumption, if we have any args preceding this it might fail
	runCmd.SetFlagErrorFunc(func(cmd *cobra.Command, err error) error {
		internal.InitConfig()
		runCmd.Run(cmd, os.Args[2:])
		return nil
	})
	RootCmd.AddCommand(liveCmd)
}

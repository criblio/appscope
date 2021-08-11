package cmd

import (
	"context"

	"github.com/criblio/scope/internal"
	"github.com/criblio/scope/relay"
	"github.com/criblio/scope/util"
	"github.com/spf13/cobra"
	"golang.org/x/sync/errgroup"
)

// relayCmd represents the relay command
var relayCmd = &cobra.Command{
	Use:   "relay",
	Short: "Data relay and aggregation utility for scoped processes",
	Long: `Data relay and aggregation utility for scoped processes

The --cribldest flag accepts a socket address in the format tcp://hostname:port, udp://hostname:port, or tls://hostname:port.`,
	Example: `scope relay -c tcp://127.0.0.1:10091
scope relay -c tls://127.0.0.1:10090`,
	Args: cobra.NoArgs,
	Run: func(cmd *cobra.Command, args []string) {
		internal.InitConfig()

		sq := relay.NewSenderQueue()

		ctx := context.Context(context.Background())
		g, gctx := errgroup.WithContext(ctx)

		g.Go(relay.Signal(gctx))
		g.Go(relay.Sender(gctx, sq))
		g.Go(relay.Receiver(gctx, sq))

		if err := g.Wait(); err != nil {
			util.ErrAndExit(err.Error())
		}
	},
}

func init() {
	relay.InitConfiguration()
	relayCmd.Flags().StringVarP(&relay.Config.CriblDest, "cribldest", "c", "", "Set Cribl destination for metrics & events (host:port defaults to tls://)")
	relayCmd.MarkFlagRequired("cribldest")
	RootCmd.AddCommand(relayCmd)
}

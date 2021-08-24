package cmd

import (
	"context"

	"github.com/criblio/scope/clients"
	"github.com/criblio/scope/internal"
	"github.com/criblio/scope/relay"
	"github.com/criblio/scope/util"
	"github.com/criblio/scope/web"
	log "github.com/sirupsen/logrus"
	"github.com/spf13/cobra"
	"golang.org/x/sync/errgroup"
)

// managerCmd represents the manager command
var managerCmd = &cobra.Command{
	Use:   "manager",
	Short: "Manage one or more scope processes",
	Long: `Manage one or more scoped processes through the web interface, and optionally relay all data to a destination using a single connection.

The --relay flag accepts a socket address in the format tcp://hostname:port, udp://hostname:port, or tls://hostname:port.`,
	Example: `scope manager -c tcp://127.0.0.1:10091
scope manager -c tls://127.0.0.1:10090`,
	Args: cobra.NoArgs,
	Run: func(cmd *cobra.Command, args []string) {
		internal.InitConfig()

		c := clients.NewClients()
		sq := relay.NewSenderQueue()
		ctx := context.Context(context.Background())
		g, gctx := errgroup.WithContext(ctx)

		g.Go(util.Signal(gctx))
		g.Go(clients.Receiver(gctx, g, sq, c))
		g.Go(web.Server(gctx, g, c))

		if relay.Config.CriblDest != "" {
			log.Info("Relay is ON with ", relay.Config.Workers, " workers and a queue size of ", relay.Config.SenderQueueSize)
			for id := 0; id < relay.Config.Workers; id++ {
				g.Go(relay.Sender(gctx, sq, id))
			}
		}

		if err := g.Wait(); err != nil {
			util.ErrAndExit(err.Error())
		}
	},
}

func init() {
	relay.InitConfiguration()
	managerCmd.Flags().StringVarP(&relay.Config.CriblDest, "relay", "c", "", "Turn on Relay and set Cribl destination for metrics & events (host:port defaults to tls://)")
	managerCmd.Flags().IntVarP(&relay.Config.Workers, "workers", "w", 1, "Set number of workers to connect and send messages to the Cribl destination")
	managerCmd.Flags().IntVarP(&relay.Config.SenderQueueSize, "queuesize", "q", 1000000, "Set buffer size for the relay (number of messages to queue)")
	managerCmd.Flags().StringVarP(&relay.Config.AuthToken, "authtoken", "a", "", "Set AuthToken for Cribl")
	RootCmd.AddCommand(managerCmd)
}

package cmd

import (
	"context"

	"github.com/criblio/scope/clients"
	"github.com/criblio/scope/internal"
	"github.com/criblio/scope/relay"
	"github.com/criblio/scope/util"
	"github.com/criblio/scope/web"
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
		g.Go(clients.Receiver(gctx, g, sq, &c))
		g.Go(web.Server(gctx, g, &c))

		if relay.Config.CriblDest != "" {
			g.Go(relay.Sender(gctx, sq))
		}

		if err := g.Wait(); err != nil {
			util.ErrAndExit(err.Error())
		}
	},
}

func init() {
	relay.InitConfiguration()
	managerCmd.Flags().StringVarP(&relay.Config.CriblDest, "relay", "c", "", "Turn on Relay and set Cribl destination for metrics & events (host:port defaults to tls://)")
	managerCmd.Flags().StringVarP(&relay.Config.AuthToken, "authtoken", "a", "", "Set AuthToken for Cribl")
	RootCmd.AddCommand(managerCmd)
}

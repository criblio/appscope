package clients

import (
	"context"

	"github.com/criblio/scope/relay"
	log "github.com/sirupsen/logrus"
	"golang.org/x/sync/errgroup"
)

// Receiver listens for new unix socket connections and manages clients
// It creates a new goroutine for each client
func Receiver(gctx context.Context, g *errgroup.Group, sq relay.Queue, c Clients) func() error {
	return func() error {

		log.Info("Receiver routine running")
		defer log.Info("Receiver routine exited")

		for {
			select {

			// case client connect:
			// accept
			// add to clients
			// clients[fd] = NewClient(socket)
			// start client worker thread:
			// g.Go clientReceiver(gctx context.Context, sq Queue)

			case <-gctx.Done():
				return gctx.Err()
			}
		}
	}
}

// clientReceiver is a dedicated receiver for a unix client
func clientReceiver(gctx context.Context, sq relay.Queue) func() error {
	return func() error {

		for {
			select {

			// case client data:
			// listen on unix domain sockets
			// first message MUST contain psm
			// if !client.Header {
			//     loadHeader()
			// }
			// push data to sq

			// case client disconnect:
			// disconnect
			// remove from clients

			case <-gctx.Done():
				return gctx.Err()
			}
		}
	}
}

package relay

import (
	"context"

	log "github.com/sirupsen/logrus"
)

// Sender makes a connection to cribldest and sends aggregated data
func Sender(gctx context.Context, sq Queue, workerId int) func() error {
	return func() error {

		log.Info("Sender routine running [Worker ", workerId, "]")
		defer log.Info("Sender routine exited [Worker ", workerId, "]")

		c, err := NewConnection()
		if err != nil {
			return err
		}

		log.Info("Connected to Cribl Destination [Worker ", workerId, "]")

		maxDepth := 0

		for {
			if len(sq) > maxDepth {
				maxDepth = len(sq)
				log.Info("Max queue depth is now: ", maxDepth, "[Worker ", workerId, "]")
			}

			select {
			case msg := <-sq:
				if err := c.Send(msg); err != nil {
					return err
				}
			case <-gctx.Done():
				return gctx.Err()
			}
		}
	}
}

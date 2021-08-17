package relay

import (
	"context"

	log "github.com/sirupsen/logrus"
)

// Sender makes a connection to cribldest and sends aggregated data
func Sender(gctx context.Context, sq Queue) func() error {
	return func() error {

		log.Info("Sender routine running")
		defer log.Info("Sender routine exited")

		c, err := NewConnection()
		if err != nil {
			return err
		}

		log.Info("Connected to Cribl Destination")

		/*
			// send a connection header on connection to the cribl destination
			header := libscope.NewHeader()
			header.Format = "ndjson"
			header.AuthToken = Config.AuthToken
			headerJson, err := json.Marshal(header)
			if err != nil {
				return err
			}
			if err = c.Send(Message(headerJson) + "\n"); err != nil {
				return err
			}
			log.Info("Sent Connection Header to Cribl Destination")
		*/

		for {
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

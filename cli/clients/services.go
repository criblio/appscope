package clients

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"

	//"fmt"
	"net"

	"github.com/criblio/scope/libscope"
	"github.com/criblio/scope/relay"
	log "github.com/sirupsen/logrus"
	"golang.org/x/sync/errgroup"
)

// Receiver listens for new unix socket connections and manages clients
// It creates a new goroutine for each client
func Receiver(gctx context.Context, g *errgroup.Group, sq relay.Queue, c *Clients) func() error {
	return func() error {

		log.Info("Receiver routine running")
		defer log.Info("Receiver routine exited")

		l, err := Listen()
		if err != nil {
			return err
		}
		defer l.Close()

		// Listen is a blocking call. The below goroutine makes it non blocking
		// It does not execute in a group context because we don't want to wait for it
		newConns := make(chan net.Conn)
		go clientListener(l, newConns)

		for {
			select {
			case conn := <-newConns:
				if conn == nil {
					return errors.New("Error in Accept")
				}
				client := c.Create(UnixConnection{conn}, libscope.Header{})
				g.Go(clientHandler(gctx, sq, client, c))

			case <-gctx.Done():
				return gctx.Err()
			}
		}
	}
}

// clientListener listens for new clients
// Since Listen is a blocking call, this can be terminated at any time
func clientListener(l net.Listener, newConns chan net.Conn) {

	log.Info("Client listener routine running")
	defer log.Info("Client listener routine exited")

	for {
		conn, err := l.Accept()
		if err != nil {
			if err.Error() == "accept unix @abstest: use of closed network connection" {
				return
			}
			log.WithFields(log.Fields{
				"err": err,
			}).Error("Accept failed")
			return
		}
		newConns <- conn
	}
}

// clientHandler is a dedicated handler for a unix client
func clientHandler(gctx context.Context, sq relay.Queue, client *Client, c *Clients) func() error {
	return func() error {

		log.Info("Handling client ", client.Id)
		defer log.Info("Stopped handling client ", client.Id)

		reader := bufio.NewReader(client.Conn.Conn)
		for {
			msg, err := ReadMessage(reader)
			if err != nil {
				if err := c.Delete(client.Id); err != nil {
					return err
				}
				return nil
			}

			if len(msg.Data) > 0 {

				// Push data to relay sender
				// sq <- relay.Message(msg)

				if client.ProcessStart.Format == "" {
					var header libscope.Header
					if err := json.Unmarshal([]byte(msg.Raw), &header); err != nil {
						log.WithFields(log.Fields{
							"err": err,
						}).Warn("Unmarshal failed")
						return nil
					}
					if header.Format == "" {
						log.Warn("No connection header received")
						return nil
					}
					if err := c.Update(client.Id, header); err != nil {
						log.WithFields(log.Fields{
							"err": err,
						}).Warn("Update failed")
						return nil
					}

					// temporary
					fmt.Println(header)
					c.PushConfig(client.Id, libscope.HeaderConfCurrent{
						Event: libscope.HeaderConfEvent{
							Enable: "false",
						},
					})
				}
			}

			select {
			case <-gctx.Done():
				return gctx.Err()
			default:
			}
		}
	}
}

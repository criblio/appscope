package live

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"net"

	"github.com/criblio/scope/libscope"
	log "github.com/sirupsen/logrus"
	"golang.org/x/sync/errgroup"
)

// Receiver listens for a new scope connection on a unix socket and handles received data
func Receiver(gctx context.Context, g *errgroup.Group, s *Scope) func() error {
	return func() error {

		log.Info("Scope receiver routine running")
		defer log.Info("Scope receiver routine exited")

		l, err := Listen()
		if err != nil {
			return err
		}
		defer l.Close()

		// Accept is a blocking call. The below goroutine makes it non blocking
		// It does not execute in a group context because we don't want to wait for it
		newConns := make(chan net.Conn)
		go scopeAccept(l, newConns)

		for {
			select {
			case conn := <-newConns:
				if conn == nil {
					return errors.New("Error in Accept")
				}
				s.Unix.Conn = conn
				g.Go(scopeHandler(gctx, s))

			case <-gctx.Done():
				return gctx.Err()
			}
		}
	}
}

// scopeAccept accepts a new scope connection
// Since Listen is a blocking call, this can be terminated at any time
func scopeAccept(l net.Listener, newConns chan net.Conn) {

	log.Info("Scope accept routine running")
	defer log.Info("Scope accept routine exited")

	for {
		conn, err := l.Accept()
		if err != nil {
			if err.Error() == "accept unix @scopelive: use of closed network connection" {
				return
			}
			log.WithFields(log.Fields{
				"err": err,
			}).Error("Accept failed")
			return
		}
		log.Info("Accepted connection from scoped process")
		newConns <- conn
	}
}

// scopeHandler is a dedicated handler for a scope connection
func scopeHandler(gctx context.Context, s *Scope) func() error {
	return func() error {

		log.Info("Handling scope")
		defer log.Info("Stopped handling scope")

		reader := bufio.NewReader(s.Unix.Conn)
		for {
			msg, err := ReadMessage(reader)
			if err != nil {
				return err
			}

			if len(msg.Data) > 0 {
				if s.ProcessStart.Format == "" {
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
					if err := s.Update(header); err != nil {
						log.WithFields(log.Fields{
							"err": err,
						}).Warn("Update failed")
						return nil
					}

					log.Info("Process Start Message received: ", header)
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

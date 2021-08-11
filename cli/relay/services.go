package relay

import (
	"context"
	"errors"
	"os"
	"os/signal"
	"syscall"

	log "github.com/sirupsen/logrus"
)

// Receiver listens for connections and buffers messages from unix sockets
func Receiver(gctx context.Context, sq Queue) func() error {
	return func() error {

		log.Info("Receiver routine running")
		defer log.Info("Receiver routine exited")

		for {
			select {

			// TODO case client:
			// accept

			// TODO case data:
			// listen on unix domain sockets
			// push everything to sq

			case <-gctx.Done():
				return gctx.Err()
			}
		}
	}
}

// Sender makes a connection to cribldest and sends aggregated data
func Sender(gctx context.Context, sq Queue) func() error {
	return func() error {

		log.Info("Sender routine running")
		defer log.Info("Sender routine exited")

		c, err := NewConnection()
		if err != nil {
			return err
		}

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

// Signal listens for SIGINT, SIGTERM, SIGQUIT
// and handles them in a graceful manner
func Signal(gctx context.Context) func() error {
	return func() error {

		log.Info("Signal routine running")
		defer log.Info("Signal routine exited")

		sig := make(chan os.Signal, 1)
		signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM, syscall.SIGQUIT)

		for {
			select {
			case s := <-sig:
				switch s {
				case syscall.SIGINT:
					log.Info("Interrupt Signal received. Beginning shutdown")
					return errors.New("Signal Detected")
				case syscall.SIGTERM:
					log.Info("Terminate Signal received. Beginning shutdown")
					return errors.New("Signal Detected")
				case syscall.SIGQUIT:
					log.Info("Quit Signal received. Beginning shutdown")
					return errors.New("Signal Detected")
				}
			case <-gctx.Done():
				return gctx.Err()
			}
		}
	}
}

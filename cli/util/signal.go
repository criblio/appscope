package util

import (
	"context"
	"errors"
	"os"
	"os/signal"
	"syscall"

	log "github.com/sirupsen/logrus"
)

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
					return errors.New("User Signal Detected")
				case syscall.SIGTERM:
					log.Info("Terminate Signal received. Beginning shutdown")
					return errors.New("User Signal Detected")
				case syscall.SIGQUIT:
					log.Info("Quit Signal received. Beginning shutdown")
					return errors.New("User Signal Detected")
				}
			case <-gctx.Done():
				return gctx.Err()
			}
		}
	}
}

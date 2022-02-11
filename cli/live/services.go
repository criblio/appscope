package live

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"net"
	"net/http"
	"time"

	"github.com/criblio/scope/libscope"
	"github.com/gin-gonic/gin"
	log "github.com/sirupsen/logrus"
	"golang.org/x/sync/errgroup"
)

func init() {
	log.SetLevel(log.DebugLevel)
}

// Receiver listens for a new scope connection on a unix socket and handles received data
// We must accept 2 connections ; one for events and metrics and one for payloads
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

		connCount := 0

		for {
			select {
			case conn := <-newConns:
				if conn == nil {
					return errors.New("Error in Accept")
				}
				s.UnixConns = append(s.UnixConns, UnixConnection{conn})
				g.Go(scopeHandler(gctx, s, connCount))
				connCount++

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
// one connection should process header, metrics, events only
// the other should process payloads only
func scopeHandler(gctx context.Context, s *Scope, connIndex int) func() error {
	return func() error {

		log.Info("Handling scope")
		defer log.Info("Stopped handling scope")

		reader := bufio.NewReader(s.UnixConns[connIndex].Conn)
		for {
			msg, err := ReadMessage(reader)
			if err != nil {
				return err
			}
			if msg != nil {
				if msg.IsHeader() && s.ProcessStart.Format == "" {
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
					} else if header.Format == "scope" {
						log.Info("Connection header received from payloads connection")
						return nil
					}
					log.Info("Connection header received from events/metrics connection")
					if err := s.Update(header); err != nil {
						log.WithFields(log.Fields{
							"err": err,
						}).Warn("Update failed")
						return nil
					}
				} else if msg.IsEvent() {
					var event libscope.Event
					if err := json.Unmarshal([]byte(msg.Raw), &event); err != nil {
						log.WithFields(log.Fields{
							"err": err,
						}).Warn("Unmarshal failed")
						return nil
					}
					s.EventBuf = append(s.EventBuf, event)
					log.Debug("Event received: ", event)
				} else if msg.IsMetric() {
					s.MetricBuf = append(s.MetricBuf, msg.Data)
					log.Debug("Metric received: ", msg.Data)
				} else if msg.IsPayload() {
					s.PayloadBuf = append(s.PayloadBuf, msg.Payload)
					log.Debug("Payload received: ", msg.Payload)
				} else {
					log.Warn("Invalid message type received")
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

// WebServer serves static web content to provide our web interface
// and provides a backend/endpoints to serve supported http requests
func WebServer(gctx context.Context, g *errgroup.Group, s *Scope) func() error {
	return func() error {

		log.Info("Web Server routine running")
		defer log.Info("Web Server routine exited")

		router := gin.Default()
		router.LoadHTMLGlob("live/templates/*")
		router.StaticFile("/favicon.ico", "./live/templates/favicon.ico")

		router.GET("/", homeHandler())
		router.GET("/api/events", scopeEventsHandler(s))
		router.GET("/api/metrics", scopeMetricsHandler(s))
		router.GET("/api/payloads", scopeConfigHandler(s))
		router.GET("/api/config", scopeConfigHandler(s))
		router.GET("/api/connections", scopeConnectionsHandler(s))
		router.GET("/api/files", scopeFilesHandler(s))

		srv := &http.Server{
			Addr:    ":8080",
			Handler: router,
		}

		g.Go(func() error {
			log.Info("Listening for HTTP requests on ", srv.Addr)
			defer log.Info("No longer listening to HTTP requests")
			if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
				return err
			}
			return nil
		})

		for {
			select {
			case <-gctx.Done():

				// The context is used to inform the server it has 2 seconds to finish
				// the request it is currently handling
				ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
				defer cancel()

				// Notify srv.ListenAndServe to shutdown
				srv.Shutdown(ctx)
				return gctx.Err()
			}
		}
	}
}

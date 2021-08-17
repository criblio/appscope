package web

import (
	"context"
	"net/http"
	"time"

	"github.com/criblio/scope/clients"
	"github.com/gin-gonic/gin"
	log "github.com/sirupsen/logrus"
	"golang.org/x/sync/errgroup"
)

// Server serves static web content to provide our web interface
// and provides a backend/endpoints to serve supported http requests
func Server(gctx context.Context, g *errgroup.Group, c clients.Clients) func() error {
	return func() error {

		log.Info("Web Server routine running")
		defer log.Info("Web Server routine exited")

		router := gin.Default()
		router.GET("/ping", func(c *gin.Context) {
			c.JSON(200, gin.H{
				"message": "pong",
			})
		})
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

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
func Server(gctx context.Context, g *errgroup.Group, c *clients.Clients) func() error {
	return func() error {

		log.Info("Web Server routine running")
		defer log.Info("Web Server routine exited")

		router := gin.Default()

		router.GET("/api/scope", scopeSearchHandler(c))
		router.GET("/api/scope/:id", scopeReadHandler(c))
		router.POST("/api/scope/:id", scopeConfigHandler(c))

		router.GET("/api/group", groupSearchHandler(c))
		router.POST("/api/group", groupCreateHandler(c))
		router.GET("/api/group/:id", groupReadHandler(c))
		router.PUT("/api/group/:id", groupUpdateHandler(c))
		router.DELETE("/api/group/:id", groupDeleteHandler(c))
		router.GET("/api/group/:id/apply", groupApplyHandler(c))

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

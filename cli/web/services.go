package web

import (
	"context"
	"net/http"
	"strconv"
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

        // search scopes
        router.GET("/api/scope", func(ctx *gin.Context) {
            items := c.Search()

            ctx.JSON(http.StatusOK, gin.H{
                "success": true,
                "len": len(items),
                "items": items,
            })
        })

        // read one scope
        router.GET("/api/scope/:id", func(ctx *gin.Context) {
			id, err := strconv.Atoi(ctx.Param("id"))
			if err != nil || id < 0 {
				ctx.JSON(http.StatusBadRequest, gin.H{
					"success": false,
					"error": "Invalid ID; " + err.Error(),
				})
				return
			}

            item, err := c.Read(uint(id))
			if err != nil {
				ctx.JSON(http.StatusNotFound, gin.H{
					"success": false,
					"error": err.Error(),
				})
				return
			}

			ctx.JSON(http.StatusOK, gin.H{
				"success": true,
				"len": 1,
				"items": [1]clients.Client{item},
			})
        })

/*
        // push config to one scope
        router.POST("/api/scope/:id", func(ctx *gin.Context) {
            var config libscope.HeaderConfig
            c.BindJSON(&config)
            c.PushConfig(ctx.Param("id"), config)
            ctx.JSON(http.StatusOK, gin.H{
                "success": true,
            })
        })

        // search groups
        router.GET("/api/group", func(ctx *gin.Context) {
            items := c.Groups.Search()
            ctx.JSON(http.StatusOK, gin.H{
                "success": true,
                "len": len(items),
                "items": items,
            })
        })

        // read one group
        router.GET("/api/group/:id", func(ctx *gin.Context) {
            items := c.Groups.Read(ctx.Param("id"))
            ctx.JSON(http.StatusOK, gin.H{
                "success": true,
                "len": len(items),
                "items": items,
            })
        })

        // update one group
        router.POST("/api/group/:id", func(ctx *gin.Context) {
            var group clients.Group
            c.BindJSON(&group)
            c.Groups.Update(ctx.Param("id"), group)
            ctx.JSON(http.StatusOK, gin.H{
                "success": true,
                "items": group, // make array
            })
        })

        //router.PUT("/api/group/:id/apply", ... push config to members)
*/
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

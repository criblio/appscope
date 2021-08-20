package web

import (
	"context"
	"net/http"
	"strconv"
	"time"

	"github.com/criblio/scope/clients"
	"github.com/criblio/scope/libscope"
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

func scopeSearchHandler(c *clients.Clients) gin.HandlerFunc {
	return func(ctx *gin.Context) {
		items := c.Search()

		ctx.JSON(http.StatusOK, gin.H{
			"success": true,
			"len":     len(items),
			"items":   items,
		})
	}
}

func scopeReadHandler(c *clients.Clients) gin.HandlerFunc {
	return func(ctx *gin.Context) {
		id, err := strconv.Atoi(ctx.Param("id"))
		if err != nil || id < 0 {
			ctx.JSON(http.StatusBadRequest, gin.H{
				"success": false,
				"error":   "Invalid ID; " + err.Error(),
			})
			return
		}

		item, err := c.Read(uint(id))
		if err != nil {
			ctx.JSON(http.StatusNotFound, gin.H{
				"success": false,
				"error":   err.Error(),
			})
			return
		}

		ctx.JSON(http.StatusOK, gin.H{
			"success": true,
			"len":     1,
			"items":   [1]clients.Client{item},
		})
	}
}

func scopeConfigHandler(c *clients.Clients) gin.HandlerFunc {
	return func(ctx *gin.Context) {
		id, err := strconv.Atoi(ctx.Param("id"))
		if err != nil || id < 0 {
			ctx.JSON(http.StatusBadRequest, gin.H{
				"success": false,
				"error":   "Invalid ID; " + err.Error(),
			})
			return
		}

		_, err = c.Read(uint(id))
		if err != nil {
			ctx.JSON(http.StatusNotFound, gin.H{
				"success": false,
				"error":   err.Error(),
			})
			return
		}

		var json libscope.HeaderConfCurrent
		if err := ctx.ShouldBindJSON(&json); err != nil {
			ctx.JSON(http.StatusBadRequest, gin.H{
				"success": false,
				"error":   err.Error(),
			})
			return
		}

		if err := c.PushConfig(uint(id), json); err != nil {
			ctx.JSON(http.StatusBadRequest, gin.H{
				"success": false,
				"error":   err.Error(),
			})
			return
		}

		ctx.JSON(http.StatusOK, gin.H{
			"success": true,
			// TODO Anything else to return here?
		})
	}
}

func groupSearchHandler(c *clients.Clients) gin.HandlerFunc {
	return func(ctx *gin.Context) {
		items := c.Groups.Search()

		ctx.JSON(http.StatusOK, gin.H{
			"success": true,
			"len":     len(items),
			"items":   items,
		})
	}
}

func groupCreateHandler(c *clients.Clients) gin.HandlerFunc {
	return func(ctx *gin.Context) {
		var json clients.Group
		if err := ctx.ShouldBindJSON(&json); err != nil {
			ctx.JSON(http.StatusBadRequest, gin.H{
				"success": false,
				"error":   err.Error(),
			})
			return
		}

		item := c.Groups.Create(json.Name, json.Config, json.Filters)

		// TODO: Add Location header to response

		ctx.JSON(http.StatusCreated, gin.H{
			"success": true,
			"len":     1,
			"items":   [1]clients.Group{*item},
		})
	}
}

func groupReadHandler(c *clients.Clients) gin.HandlerFunc {
	return func(ctx *gin.Context) {
		id, err := strconv.Atoi(ctx.Param("id"))
		if err != nil || id < 0 {
			ctx.JSON(http.StatusBadRequest, gin.H{
				"success": false,
				"error":   "Invalid ID; " + err.Error(),
			})
			return
		}

		item, err := c.Groups.Read(uint(id))
		if err != nil {
			ctx.JSON(http.StatusNotFound, gin.H{
				"success": false,
				"error":   err.Error(),
			})
			return
		}

		ctx.JSON(http.StatusOK, gin.H{
			"success": true,
			"len":     1,
			"items":   [1]clients.Group{*item},
		})
	}
}

func groupUpdateHandler(c *clients.Clients) gin.HandlerFunc {
	return func(ctx *gin.Context) {
		id, err := strconv.Atoi(ctx.Param("id"))
		if err != nil || id < 0 {
			ctx.JSON(http.StatusBadRequest, gin.H{
				"success": false,
				"error":   "Invalid ID; " + err.Error(),
			})
			return
		}

		item, err := c.Groups.Read(uint(id))
		if err != nil {
			ctx.JSON(http.StatusNotFound, gin.H{
				"success": false,
				"error":   err.Error(),
			})
			return
		}

		var json clients.Group
		if err := ctx.ShouldBindJSON(&json); err != nil {
			ctx.JSON(http.StatusBadRequest, gin.H{
				"success": false,
				"error":   err.Error(),
			})
			return
		}

		// TODO: need a way to update item and save it instead of risking
		// partial updates and having to re-Read at the end.
		if err := c.Groups.Update(uint(id), json.Name); err != nil {
			ctx.JSON(http.StatusBadRequest, gin.H{
				"success": false,
				"error":   err.Error(),
			})
			return
		}
		if err := c.Groups.Update(uint(id), json.Config); err != nil {
			ctx.JSON(http.StatusBadRequest, gin.H{
				"success": false,
				"error":   err.Error(),
			})
			return
		}
		if err := c.Groups.Update(uint(id), json.Filters); err != nil {
			ctx.JSON(http.StatusBadRequest, gin.H{
				"success": false,
				"error":   err.Error(),
			})
			return
		}
		item, _ = c.Groups.Read(uint(id))

		ctx.JSON(http.StatusOK, gin.H{
			"success": true,
			"len":     1,
			"items":   [1]clients.Group{*item},
		})
	}
}

func groupDeleteHandler(c *clients.Clients) gin.HandlerFunc {
	return func(ctx *gin.Context) {
		id, err := strconv.Atoi(ctx.Param("id"))
		if err != nil || id < 0 {
			ctx.JSON(http.StatusBadRequest, gin.H{
				"success": false,
				"error":   "Invalid ID; " + err.Error(),
			})
			return
		}

		item, err := c.Groups.Read(uint(id))
		if err != nil {
			ctx.JSON(http.StatusNotFound, gin.H{
				"success": false,
				"error":   err.Error(),
			})
			return
		}

		if err := c.Groups.Delete(uint(id)); err != nil {
			ctx.JSON(http.StatusNotFound, gin.H{
				"success": false,
				"error":   err.Error(),
			})
			return
		}

		ctx.JSON(http.StatusOK, gin.H{
			"success": true,
			"len":     1,
			"items":   [1]clients.Group{*item},
		})
	}
}

func groupApplyHandler(c *clients.Clients) gin.HandlerFunc {
	return func(ctx *gin.Context) {
		id, err := strconv.Atoi(ctx.Param("id"))
		if err != nil || id < 0 {
			ctx.JSON(http.StatusBadRequest, gin.H{
				"success": false,
				"error":   "Invalid ID; " + err.Error(),
			})
			return
		}

		_, err = c.Groups.Read(uint(id))
		if err != nil {
			ctx.JSON(http.StatusNotFound, gin.H{
				"success": false,
				"error":   err.Error(),
			})
			return
		}
		/*
			if err := c.PushGroupConfig(id); err != nil {
				ctx.JSON(http.StatusInternalServerError, gin.H{
					"success": false,
					"error": err.Error(),
				})
				return
			}
		*/
		ctx.JSON(http.StatusOK, gin.H{
			"success": true,
			// TODO list scopes that were applied? The new config?
		})
	}
}

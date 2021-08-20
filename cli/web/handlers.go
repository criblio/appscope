package web

import (
	"net/http"
	"strconv"

	"github.com/criblio/scope/clients"
	"github.com/criblio/scope/run"
	"github.com/gin-gonic/gin"
)

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

		var json run.ScopeConfig
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

		item := c.Groups.Create(json.Name, json.Config, json.Filter)

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
		if err := c.Groups.Update(uint(id), json.Filter); err != nil {
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

		if err := c.PushGroupConfig(uint(id)); err != nil {
			ctx.JSON(http.StatusInternalServerError, gin.H{
				"success": false,
				"error":   err.Error(),
			})
			return
		}

		ctx.JSON(http.StatusOK, gin.H{
			"success": true,
			// TODO list scopes that were applied? The new config?
		})
	}
}

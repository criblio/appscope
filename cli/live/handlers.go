package live

import (
	"net/http"

	"github.com/gin-gonic/gin"
)

// TODO thread safety for s

func scopeEventsHandler(s *Scope) gin.HandlerFunc {
	return func(ctx *gin.Context) {
		ctx.JSON(http.StatusOK, gin.H{
			"success": true,
			"len":     1,
			"items":   s.EventBuf,
		})
	}
}

func scopeMetricsHandler(s *Scope) gin.HandlerFunc {
	return func(ctx *gin.Context) {
		ctx.JSON(http.StatusOK, gin.H{
			"success": true,
			"len":     1,
			"items":   s.MetricBuf,
		})
	}
}

func scopePayloadsHandler(s *Scope) gin.HandlerFunc {
	return func(ctx *gin.Context) {
		ctx.JSON(http.StatusOK, gin.H{
			"success": true,
			"len":     1,
			"items":   s.PayloadBuf,
		})
	}
}

func scopeConfigHandler(s *Scope) gin.HandlerFunc {
	return func(ctx *gin.Context) {
		ctx.JSON(http.StatusOK, gin.H{
			"success": true,
			"len":     1,
			"items":   s.ProcessStart.Info,
		})
	}
}

func scopeConnectionsHandler(s *Scope) gin.HandlerFunc {
	return func(ctx *gin.Context) {
		ctx.JSON(http.StatusOK, gin.H{
			"success": true,
			"len":     1,
			"items":   s, // TODO
		})
	}
}

package live

import (
	"encoding/json"
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"
)

func homeHandler() gin.HandlerFunc {
	return func(ctx *gin.Context) {
		ctx.HTML(
			http.StatusOK,
			"index.html",
			gin.H{
				"title": "AppScope Live",
			},
		)
	}
}

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

		type connection struct {
			NetType string
			MessageData
		}
		connections := []connection{}
		for i, e := range s.EventBuf {
			bytes, _ := json.Marshal(e)
			event := string(bytes)
			if strings.Contains(event, "net.open") {
				connections = append(connections, connection{
					"net.open",
					s.EventBuf[i],
				})
			} else if strings.Contains(event, "net.close") {
				connections = append(connections, connection{
					"net.close",
					s.EventBuf[i],
				})
			}
		}

		ctx.JSON(http.StatusOK, gin.H{
			"success": true,
			"len":     1,
			"items":   connections,
		})
	}
}

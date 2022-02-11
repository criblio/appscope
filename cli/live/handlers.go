package live

import (
	"net/http"

	"github.com/gin-gonic/gin"
)

func homeHandler() gin.HandlerFunc {
	return func(ctx *gin.Context) {
		ctx.HTML(
			http.StatusOK,
			"index.html",
			gin.H{
				"title": "AppScope",
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

		// send an array of http target URLs
		var connections []string

		for _, event := range s.EventBuf {
			if event.Body.SourceType == "http" {
				connection := event.Body.Data.HttpHost + event.Body.Data.HttpTarget
				found := false
				for _, c := range connections {
					if connection == c {
						found = true
						break
					}
				}
				if !found {
					connections = append(connections, connection)
				}
			}
		}

		ctx.JSON(http.StatusOK, gin.H{
			"success": true,
			"len":     1,
			"items":   connections,
		})
	}
}

func scopeFilesHandler(s *Scope) gin.HandlerFunc {
	return func(ctx *gin.Context) {

		// send an array of files written to
		var files []string

		for _, event := range s.EventBuf {
			if event.Body.Data.FileWriteBytes > 0 {
				file := event.Body.Data.File
				found := false
				for _, f := range files {
					if file == f {
						found = true
						break
					}
				}
				if !found {
					files = append(files, file)
				}
			}
		}

		ctx.JSON(http.StatusOK, gin.H{
			"success": true,
			"len":     1,
			"items":   files,
		})
	}
}

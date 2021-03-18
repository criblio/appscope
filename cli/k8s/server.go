package k8s

import (
	"fmt"
	"net/http"

	"github.com/rs/zerolog/log"
)

// StartServer starts the server
func (opt Options) StartServer() error {
	app := &App{&opt}

	http.HandleFunc("/mutate", app.HandleMutate)

	log.Info().Msgf("Listening on port %d\n", opt.Port)

	return http.ListenAndServeTLS(fmt.Sprintf(":%d", opt.Port), app.CertFile, app.KeyFile, nil)
}

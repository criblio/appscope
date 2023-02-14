package daemon

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"net"
	"path/filepath"
	"strings"
	"time"

	"github.com/rs/zerolog/log"
)

// Connect establishes a tcp connection to filedest
func (d *Daemon) Connect() error {
	if d.filedest == "" {
		log.Error().Msgf("error no filedest specified")
		return errors.New("error no filedest specified")
	}

	var err error
	if d.connection, err = net.Dial("tcp", d.filedest); err != nil {
		if _, t := err.(*net.OpError); t {
			log.Error().Err(err).Msgf("error connecting to %s", d.filedest)
		} else {
			log.Error().Err(err).Msgf("unknown error connecting to %s", d.filedest)
		}
		return err
	}

	return nil
}

// Disconnect disconnects the filedest tcp connection
func (d *Daemon) Disconnect() error {
	return d.connection.Close()
}

// SendFiles attempts to send files to the daemon network destination
// The useJson argument allows the user to embed the file contents in a json object
func (d *Daemon) SendFiles(filepaths []string, useJson bool) error {
	for _, f := range filepaths {
		var msg []byte

		// Read file from dir
		fileBytes, err := ioutil.ReadFile(f)
		if err != nil {
			log.Error().Err(err).Msgf("error reading file %s", f)
			continue
		}

		// Embed the file contents in a json object if requested
		if useJson {
			fileName := filepath.Base(f)

			// Create a JSON data structure from the data
			dataMap := make(map[string]interface{})
			var data interface{}
			if err := json.Unmarshal(fileBytes, &data); err != nil {
				// If file data is not json
				dataMap[fileName] = strings.Split(string(fileBytes), "\n")
			} else {
				// If file data is json
				dataMap[fileName] = data
			}

			// Convert the data structure to a JSON string
			jsonData, err := json.Marshal(dataMap)
			if err != nil {
				fmt.Println("Error converting to JSON:", err)
				return err
			}

			msg = jsonData
		} else {
			msg = fileBytes
		}

		// Write data to daemon tcp connection
		d.connection.SetWriteDeadline(time.Now().Add(1 * time.Second))
		if _, err = d.connection.Write(msg); err != nil {
			log.Error().Err(err).Msgf("error writing to %s", d.filedest)
			return err
		}
		if _, err = d.connection.Write([]byte("\n")); err != nil {
			log.Error().Err(err).Msgf("error writing to %s", d.filedest)
			return err
		}
	}

	return nil
}

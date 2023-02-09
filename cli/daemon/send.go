package daemon

import (
	"errors"
	"io/ioutil"
	"net"
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
func (d *Daemon) SendFiles(filepaths []string) error {
	for _, f := range filepaths {
		// Read file from dir
		fileBytes, err := ioutil.ReadFile(f)
		if err != nil {
			log.Error().Err(err).Msgf("error reading file %s", f)
			continue
		}

		// Write data to daemon tcp connection
		d.connection.SetWriteDeadline(time.Now().Add(1 * time.Second))
		if _, err = d.connection.Write(fileBytes); err != nil {
			log.Error().Err(err).Msgf("error writing to %s", d.filedest)
			return err
		}
	}

	return nil
}

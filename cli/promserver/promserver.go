package promserver

import (
	"net/http"
	"fmt"
	"path/filepath"
	"os"
	"io"
	"net"
	"log"
	"bytes"
	str "strings"
)

/*
 * This represents a handler that runs in response to an
 * HTTP GET /metrics request, from a Prometheus scraper.
 *
 * Metric data is obtained from a go routine that reads
 * from a TCP connection used by libscope apps. It is
 * expected that the metrics destination for scoped
 * apps uses TCP with the mport value.
 */

// Set to > 0 to enable console messages
var msgs = 1

func pmsg(msg ...interface{}) {
     if msgs != 0 {
        for _, pvar:= range msg {
            fmt.Print(pvar, " ")
        }
        fmt.Println()
     }
}

/*
 * Read metrics sent from libscope
 * Store in a temp file
 */
func Metrics(conn net.Conn) {
	defer conn.Close()

	fname, err := os.CreateTemp("", "scope-metrics-*")
    if err != nil {
        log.Fatal("Open temp file:", err)
        return
    }

	defer fname.Close()

	metrics, err := io.ReadAll(conn)
	if err != nil {
		log.Fatal(err)
		return
	}

	_, err = fname.Write(metrics);
	if err != nil {
		log.Fatal("Write:", err)
		return
	}
}

/*
 * Gather all metrics data and return an HTTP response
 */
func Handler(rwrite http.ResponseWriter, req *http.Request) {
	tpath:= os.TempDir()
	pmsg("checking dir: ", tpath)

	dir, err := os.ReadDir(tpath)
	if err != nil {
        fmt.Println(err)
		return
    }

	// create a new buffer with an initial size of 0
    buf := new(bytes.Buffer)

	for _, entry := range dir {
		currf := entry.Name()
		mode := entry.Type()

		if mode.IsRegular() && str.HasPrefix(currf, "scope-metrics-") {
			pmsg("Found metrics file: ", currf)

			mpath := filepath.Join(tpath, currf)
			mfile, err := os.Open(mpath)
			if err != nil {
				continue
			}

			nbytes, err := buf.ReadFrom(mfile)
			if err != nil {
				pmsg("ERROR: read from: ", mpath)
			} else {
				pmsg("Read ", nbytes, " from ", mfile, " buf len: ", buf.Len())
			}

			mfile.Close()
			os.Remove(mpath)
		}
	}

	rwrite.Header().Set("Content-Type", "text/plain")
	rwrite.Write(buf.Bytes())
}

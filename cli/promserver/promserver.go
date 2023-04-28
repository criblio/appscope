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

const prefix = "scope-metrics-"

// Set to > 0 to enable console messages
var msgs = 1

// List of files that can be deleted; the remote socket is closed
var delList = make(map[string]bool)

func Pmsg(msg ...interface{}) {
     if msgs != 0 {
        for _, pvar:= range msg {
            fmt.Print(pvar, " ")
        }
        fmt.Println()
     }
}

func GetMfiles() []string {
	var mfiles[]string

	tpath:= os.TempDir()
	dir, err := os.ReadDir(tpath)
	if err != nil {
		fmt.Println(err)
		return nil
	}

	for _, entry := range dir {
		currf := entry.Name()
		mode := entry.Type()

		if mode.IsRegular() && str.HasPrefix(currf, prefix) {
			mpath := filepath.Join(tpath, currf)
			mfiles = append(mfiles, mpath)
			fmt.Println("getMfile adding file: ", mpath)
		}
	}

	return mfiles
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

	/*
     * Note: close this here because we open, write, close
     * in the loop below. This enables the metric file to
     * be truncated and the new len written each time new
     * metric data is received.
     */
	fname.Close()

	Pmsg("Metrics: ", fname.Name())

	metrics := make([]byte, 1024)

	for  {
		_, err := conn.Read(metrics)
		if err != nil || err == io.EOF || len(metrics) == 0 {
			log.Println(err)
			break
		}

		tfile, err := os.OpenFile(fname.Name(), os.O_RDWR|os.O_APPEND, 0666)
		if err != nil {
			log.Fatal("Open temp file:", err)
			return
		}

		_, err = tfile.Write(metrics);
		if err != nil {
			// Continue to read more?
			log.Println("Write:", err)
		}

		tfile.Close()
	}

	Pmsg("Adding ", fname.Name(), " to the del list")
	delList[fname.Name()] = true
}

/*
 * Gather all metrics data and return an HTTP response
 */
func Handler(rwrite http.ResponseWriter, req *http.Request) {
	// create a new buffer with an initial size of 0
    buf := new(bytes.Buffer)

	mfiles := GetMfiles()
	for _, mpath := range mfiles {
		mfile, err := os.OpenFile(mpath, os.O_RDWR, 0666)
		if err != nil {
			Pmsg("ERROR: open of: ", mpath)
			continue
		}

		nbytes, err := buf.ReadFrom(mfile)
		if err != nil {
			Pmsg("ERROR: read from: ", mpath)
		} else {
			Pmsg("Read ", nbytes, " from ", mpath, " buf len: ", buf.Len())
		}

		mfile.Truncate(0)
		mfile.Close()
	}

	// Send an HTTP response
	rwrite.Header().Set("Content-Type", "text/plain")
	rwrite.Write(buf.Bytes())

	// Remove any files from processes that have exited
	for name, enable := range delList {
		Pmsg("Removing: ", name, " enabled: ", enable)
		os.Remove(name)
	}
}

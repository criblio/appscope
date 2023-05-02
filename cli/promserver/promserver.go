package promserver

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"path/filepath"
	str "strings"
	"syscall"
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
		for _, pvar := range msg {
			fmt.Print(pvar, " ")
		}
		fmt.Println()
	}
}

func GetMfiles() []string {
	var mfiles []string

	tpath := os.TempDir()
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

	// A bufio reader supports reading lines from the socket
	reader := bufio.NewReader(conn)

	for {
		// Read lines from the connection
		line, err := reader.ReadString('\n')
		if err != nil {
			log.Println("Error reading from connection:", err)
			break
		}

		tfile, err := os.OpenFile(fname.Name(), os.O_RDWR|os.O_APPEND, 0666)
		if err != nil {
			log.Fatal("Open temp file:", err)
			return
		}

		fd := int(tfile.Fd())
		_ = syscall.Flock(fd, syscall.LOCK_EX)

		_, err = tfile.WriteString(line)
		if err != nil {
			// Continue to read more?
			log.Println("Write:", err)
		}

		_ = syscall.Flock(fd, syscall.LOCK_UN)
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

		fd := int(mfile.Fd())
		_ = syscall.Flock(fd, syscall.LOCK_EX)

		nbytes, err := buf.ReadFrom(mfile)
		if err != nil {
			Pmsg("ERROR: read from: ", mpath)
		} else {
			Pmsg("Read ", nbytes, " from ", mpath, " buf len: ", buf.Len())
		}

		mfile.Truncate(0)
		_ = syscall.Flock(fd, syscall.LOCK_UN)
		mfile.Close()
	}

	// Send an HTTP response
	rwrite.Header().Set("Content-Type", "text/plain; charset=utf-8")
	rwrite.WriteHeader(http.StatusOK)
	io.WriteString(rwrite, string(buf.Bytes()))

	// Remove any files from processes that have exited
	for name, enable := range delList {
		Pmsg("Removing: ", name, " enabled: ", enable)
		os.Remove(name)
	}
}

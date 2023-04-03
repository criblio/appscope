package promserver

import (
	"net/http"
	"fmt"
	"path/filepath"
	"os"
)

/*
 * This represents the handler that runs in response to an
 * HTTP GET /metrics request, from a Prometheus scraper.
 *
 * The primary use case is that of a k8s cluster.
 * It is expected that this executes from a container with a
 * hostfs read-only mount point to the host FS.
 *
 * The handler searches /hostfs/proc/<pid> for a root/scope/metrics
 * entry. Where /scope, at a k8s pod level, is defined by a
 * scope k8s command. It then walks subdirs intended to include
 * process specific metrics. An HTTP response is sent with all
 * available metrics. The intent is that all application metrics
 * emitted from libscope for all scoped apps will be accessible.
 * 
 * The paths used are currently hard coded. At least the start point
 * for a search should probably be passed as command line options.
 * The specific subdirs and file name must be compatible with
 * libscope behavior, yet to be defined at this point.
 * example: /hostfs/proc/4693/root/scope/metrics/1/metrics.json
 */

const pelement1 = "/hostfs"
const pelement2 = "proc/"
const pelement3 = "root/"
const pelement4 = "scope/"
const pelement5 = "metrics/"
const pelement6 = "metrics.json"

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

func faccess(path string) (bool) {
    if _, err := os.Stat(path); err != nil {
       return false
    }

    return true
}

func Handler(rwrite http.ResponseWriter, req *http.Request) {
	path := filepath.Join(pelement1, pelement2)
    pmsg("checking dir: ", path)
	dir, err := os.ReadDir(path)
	if err != nil {
        fmt.Println(err)
		return
    }

	path = filepath.Join(os.TempDir(), "appm")
	tempf, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_RDWR, 0644)
	if err != nil {
		fmt.Println(err)
		return
	}

	defer tempf.Close()
	defer os.Remove(tempf.Name())

	for _, entry := range dir {
		if entry.IsDir() {
			dname := entry.Name()

            // /hostfs/proc/dname/root/scope/metrics/
			path := filepath.Join(pelement1, pelement2, dname,	pelement3, pelement4, pelement5)
            if faccess(path) == true {
               metricd, err := os.Open(path)
               if err != nil {
                  fmt.Println(err)
                  return
               }

               mdname, _ := metricd.Readdirnames(0)
               for _, ename := range mdname {
                     // /hostfs/proc/dname/root/scope/metrics/ename/metrics.json
                     path := filepath.Join(pelement1, pelement2, dname,	pelement3,
                          pelement4, pelement5, ename, pelement6)

                     metricf, err := os.Open(path)
                     if err != nil {
                        continue
			         }

                     pmsg("found pid: ", dname, "file: ", path)

			         metrici, err := metricf.Stat()
                     if err != nil {
				        continue
			         }

                     metricbuf := make([]byte, metrici.Size())
			         metricd, err := metricf.Read(metricbuf)
			         if err != nil {
				        continue
			         }

                     pmsg("read ", metricd, " bytes")

			         if _, err := tempf.Write(metricbuf); err != nil {
				        continue
			         }

			         metricf.Close()
		       }

               metricd.Close()
            }
        }
    }

	if _, err := tempf.Seek(0, 0); err != nil {
		fmt.Println(err)
		return
	}

	tempi, err := tempf.Stat()
	tempbuf := make([]byte, tempi.Size())
	tempd, err := tempf.Read(tempbuf)
	if err != nil {
		fmt.Println(err)
		rwrite.Write([]byte("This is an example server.\n"))
		return
	}

    pmsg("read ", tempd, " bytes", "temp size: ", tempi.Size())

    rwrite.Header().Set("Content-Type", "text/plain")
	rwrite.Write(tempbuf)

}

package promserver

import (
	"net/http"
	"fmt"
	"path/filepath"
	"os"
)

const pelement1 = "/hostfs"
const pelement2 = "proc/"
const pelement3 = "root/"
const pelement4 = "scope/"
const pelement5 = "metrics/"
const pelement6 = "metrics.json"

func faccess(path string) (bool) {
    if _, err := os.Stat(path); err != nil {
       return false
    }

    return true
}

func Handler(rwrite http.ResponseWriter, req *http.Request) {
	path := filepath.Join(pelement1, pelement2)
	fmt.Println("checking dir: ", path)
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
                     // /hostfs/proc/dname/root/scope/metrics/mdname/metrics.json
                     path := filepath.Join(pelement1, pelement2, dname,	pelement3,
                          pelement4, pelement5, ename, pelement6)

                     fmt.Println(err, path, mdname)
                     metricf, err := os.Open(path)
                     if err != nil {
                        continue
			         }

			         fmt.Println("found pid: ", dname, "file: ", path)

			         metrici, err := metricf.Stat()
                     if err != nil {
				        continue
			         }

                     metricbuf := make([]byte, metrici.Size())
			         metricd, err := metricf.Read(metricbuf)
			         if err != nil {
				        continue
			         }

			         fmt.Println("read ", metricd, " bytes")

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

	fmt.Println("read ", tempd, " bytes", "temp size: ", tempi.Size())

    rwrite.Header().Set("Content-Type", "text/plain")
	rwrite.Write(tempbuf)

}

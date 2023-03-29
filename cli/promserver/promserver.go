package promserver

import (
	"net/http"
	"fmt"
	"path/filepath"
	"os"
)

const pelement1 = "/"
const pelement2 = "proc/"
const pelement3 = "root/"
const pelement4 = "tmp/"
const pelement5 = "metrics/"
const pelement6 = "metrics.json"

func Handler(rwrite http.ResponseWriter, req *http.Request) {
	path := filepath.Join(pelement1, pelement2)
	fmt.Println("checking dir: ", path)
	dir, err := os.ReadDir(path)
	if err != nil {
        fmt.Println("Error readdir: ", path)
		return
    }

	path = filepath.Join(os.TempDir(), "appm")
	tempf, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_RDWR, 0644)
	if err != nil {
		fmt.Println("Error open temp file: ", path)
		return
	}

	defer tempf.Close()
	defer os.Remove(tempf.Name())

	for _, entry := range dir {
		if entry.IsDir() {
			dname :=entry.Name()

			path := filepath.Join(pelement1, pelement2, dname,	pelement3,
				pelement4, pelement5, dname, pelement6)

			metricf, err := os.Open(path)
			if err != nil {
				continue
			}

			fmt.Println("found pid: ", dname, "file: ", path)

			metrici, err := metricf.Stat()
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
    }

	if _, err := tempf.Seek(0, 0); err != nil {
		fmt.Println("Error seek")
		return
	}

	tempi, err := tempf.Stat()
	tempbuf := make([]byte, tempi.Size())
	tempd, err := tempf.Read(tempbuf)
	if err != nil {
		fmt.Println("Error read results")
		rwrite.Write([]byte("This is an example server.\n"))
		return
	}

	fmt.Println("read ", tempd, " bytes", "temp size: ", tempi.Size())

    rwrite.Header().Set("Content-Type", "text/plain")
	rwrite.Write(tempbuf)

}

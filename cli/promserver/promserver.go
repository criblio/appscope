package promserver

import (
	"net/http"
	"fmt"

	"github.com/criblio/scope/util"
)

func Handler(rwrite http.ResponseWriter, req *http.Request) {
	procs, err := util.ProcessesScoped()
	if err != nil {
		util.ErrAndExit("Unable to retrieve scoped processes: %v", err)
	}

    rwrite.Header().Set("Content-Type", "text/plain")
    rwrite.Write([]byte("This is an example server.\n"))

	fmt.Println(len(procs))
	for i := 0; i < len(procs); i++ {
		pid := procs[i].Pid
		fmt.Println(pid)
    }
}

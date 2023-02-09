package daemon

import (
	"net"
)

type Daemon struct {
	filedest   string
	connection net.Conn
}

func New(filedest string) Daemon {
	return Daemon{
		filedest: filedest,
	}
}

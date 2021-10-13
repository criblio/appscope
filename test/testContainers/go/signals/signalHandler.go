package main

import (
	"fmt"
	_ "net"
	"os"
	"os/signal"
	"syscall"
)

func main() {

	sig_channel := make(chan os.Signal, 1)
	quit_channel := make(chan bool, 1)

	signal.Notify(sig_channel)
	fmt.Println("awaiting for SIGINT signal")
	go func() {
		for {
			s := <-sig_channel
			fmt.Println("Got signal:", s)
			if s == syscall.SIGINT {
				quit_channel <- true
			}
		}
	}()
	<-quit_channel
	fmt.Println("exiting")
}

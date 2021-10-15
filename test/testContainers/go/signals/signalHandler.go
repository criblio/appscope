package main

import (
	"fmt"
	"log"
	_ "net"
	"os"
	"os/signal"
	"syscall"
)

func genSignalLoop(pid int) {
	for {
		syscall.Kill(pid, syscall.SIGCHLD)
	}
}

func genOpenCloseLoop(f_name string) {
	for {
		f, err := os.Open(f_name)
		if err != nil {
			log.Fatal(err)
		}
		f.Close()
	}
}

func main() {

	sig_channel := make(chan os.Signal, 1)
	quit_channel := make(chan bool, 1)
	current_pid := os.Getpid()
	exec_name, err := os.Executable()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("%s handling following signals: %d, %d, %d\n", exec_name, syscall.SIGCHLD, syscall.SIGINT, syscall.SIGUSR1)
	fmt.Printf("Current PID: %d\n", current_pid)

	tmpfile, err := os.CreateTemp("", "test_file")
	if err != nil {
		log.Fatal(err)
	}
	defer os.Remove(tmpfile.Name())

	fmt.Println("Awaiting for SIGINT signal")
	signal.Notify(sig_channel, syscall.SIGCHLD, syscall.SIGINT, syscall.SIGUSR1)
	go func() {
		for {
			s := <-sig_channel
			if s == syscall.SIGINT {
				quit_channel <- true
			}
		}
	}()

	go func() {
		genSignalLoop(current_pid)
	}()

	go func() {
		genOpenCloseLoop(tmpfile.Name())
	}()

	<-quit_channel
	fmt.Println("Exiting")
}

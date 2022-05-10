package main

import (
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"os/signal"
	"syscall"
)

const LoopLimit = 1e6

func genSignalLoop(pid int) {
	for i := 1; i < LoopLimit; i++ {
		syscall.Kill(pid, syscall.SIGCHLD)
	}
}

func genOpenCloseLoop(f_name string) {
	for i := 1; i < LoopLimit; i++ {
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
	fmt.Printf("%s handling following signals: %d, %d, %d\n", exec_name, syscall.SIGCHLD, syscall.SIGINT, syscall.SIGUSR2)
	fmt.Printf("Current PID: %d\n", current_pid)

	tmpfile, err := ioutil.TempFile("", "test_file")
	if err != nil {
		log.Fatal(err)
	}
	defer os.Remove(tmpfile.Name())

	fmt.Println("Awaiting for SIGINT signal")
	signal.Notify(sig_channel, syscall.SIGCHLD, syscall.SIGINT, syscall.SIGUSR2)
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

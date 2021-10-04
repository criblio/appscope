package main

import (
    "fmt"
    "os"
    "sync"
)

const basedir string = "/dev/shm/outdir"
const numThreads int = 100000

func check(e error) {
    if e != nil {
        panic(e)
    }
}

func out_func(wg *sync.WaitGroup, id int) {
    defer wg.Done()

    // Create new directory
    dirname := fmt.Sprintf("%s/dir%06d", basedir, id)
    err := os.Mkdir(dirname, 0777)
    check(err)

    // Create file
    filename := dirname + "/file"
    f, err := os.Create(filename)
    check(err)
    defer f.Close()

    // Write something to that file
    filedata := fmt.Sprintf("hey %06d\n", id)
    f.WriteString(filedata)
}

func main() {
    fmt.Println("Starting fileThread.go")

    // Delete contents of old basedir
    _, err := os.Stat(basedir)
    if !os.IsNotExist(err) {
        fmt.Printf("Deleting old %s directory.\n", basedir)
        os.RemoveAll(basedir)
    }

    fmt.Printf("Creating new %s directory.\n", basedir)
    os.Mkdir(basedir, 0777)

    fmt.Printf("Creating %d goroutines to create dirs and files under %s\n", numThreads, basedir)
    var wg sync.WaitGroup
    for i := 0; i < numThreads; i++ {
        wg.Add(1)
        go out_func(&wg, i)
    }

    wg.Wait()
    fmt.Println("Ending fileThread.go")
}


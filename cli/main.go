package main

/*
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sched.h>
#include <getopt.h>
#include "../src/loader/loader.h"

// long aliases for short options
static struct option opts[] = {
    {"loader",    no_argument,    0,  'l'},
    {0,           0,              0,   0 }
};

__attribute__((constructor)) void enter_namespace(int argc, char **argv) {
	int index;
    for (;;) {
		index = 0;
        int opt = getopt_long(argc, argv, "+:l", opts, &index);
        if (opt == -1) {
            break;
        }
        switch (opt) {
            case 'l':
                loader(argc, argv);
				exit(EXIT_SUCCESS);
            default:
				break;
        }
    }
}
*/
import "C"
import (
	"github.com/criblio/scope/cmd"
	"github.com/criblio/scope/internal"
)

// GitSummary is produced by govvv and stores tag, commit and branch status
var GitSummary string

// BuildDate is set by govvv and stores when we were built
var BuildDate string

// Version is set by govvv and stores the version at build time
var Version string

func main() {
	internal.SetVersionInfo(GitSummary, BuildDate, Version)
	cmd.Execute()
}

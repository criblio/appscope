package main

/*
#if defined(__aarch64__)
#cgo LDFLAGS: -L../lib/linux/aarch64 -lloader
#else
#cgo LDFLAGS: -L../lib/linux/x86_64 -lloader
#endif
#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include "../src/loader/loader.h"

// long aliases for short options
static struct option opts[] = {
    { "usage",       no_argument,       0, 'u'},
    { "help",        optional_argument, 0, 'h' },
    { "attach",      required_argument, 0, 'a' },
    { "detach",      required_argument, 0, 'd' },
    { "namespace",   required_argument, 0, 'n' },
    { "configure",   required_argument, 0, 'c' },
    { "unconfigure", no_argument,       0, 'w' },
    { "service",     required_argument, 0, 's' },
    { "unservice",   no_argument,       0, 'v' },
    { "libbasedir",  required_argument, 0, 'l' },
    { "patch",       required_argument, 0, 'p' },
    { "starthost",   no_argument,       0, 'r' },
    { "stophost",    no_argument,       0, 'x' },
    { "loader",      no_argument,       0, 'z' },
    { 0, 0, 0, 0 }
};

__attribute__((constructor)) void enter_namespace(int argc, char **argv, char **env) {
	int index;
    for (;;) {
		index = 0;
        int opt = getopt_long(argc, argv, "+:uh:a:d:n:l:f:p:c:s:rz", opts, &index);
        if (opt == -1) {
            break;
        }
        switch (opt) {
            case 'z':
                loader(argc, argv, env);
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

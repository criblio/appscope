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

// Example Usage:
// scope [OPTIONS] --ldattach PID
// scope [OPTIONS] --lddetach PID
// scope [OPTIONS] --configure FILTER_PATH --namespace PID
// scope [OPTIONS] --service SERVICE --namespace PID
// scope [OPTIONS] --passthrough EXECUTABLE [ARGS...]
// scope [OPTIONS] --patch SO_FILE
//
// Options:
// -l, --libbasedir DIR         specify parent for the library directory (default: /tmp)
// -f DIR                       alias for \"-l DIR\" for backward compatibility
// -a, --ldattach PID           attach to the specified process ID
// -d, --lddetach PID           detach from the specified process ID
// -c, --configure FILTER_PATH  configure scope environment with FILTER_PATH
// -w, --unconfigure            unconfigure scope environment
// -s, --service SERVICE        setup specified service NAME
// -v, --unservice              remove scope from all service configurations
// -n  --namespace PID          perform service/configure operation on specified container PID
// -p, --patch SO_FILE          patch specified libscope.so
// -r, --starthost              execute the scope start command in a host context (must be run in the container)
// -x, --stophost               execute the scope stop command in a host context (must be run in the container)

// Long aliases for short options
// NOTE: These must not conflict with the cli options specified in the cmd/ package
static struct option opts[] = {
    { "ldattach",    required_argument, 0, 'a' },
    { "lddetach",    required_argument, 0, 'd' },
    { "namespace",   required_argument, 0, 'n' },
    { "configure",   required_argument, 0, 'c' },
    { "unconfigure", no_argument,       0, 'w' },
    { "service",     required_argument, 0, 's' },
    { "unservice",   no_argument,       0, 'v' },
    { "libbasedir",  required_argument, 0, 'l' },
    { "patch",       required_argument, 0, 'p' },
    { "starthost",   no_argument,       0, 'r' },
    { "stophost",    no_argument,       0, 'x' },
    { "passthrough", no_argument,       0, 'z' },
    { 0, 0, 0, 0 }
};

__attribute__((constructor)) void cli_constructor(int argc, char **argv, char **env) {
	bool opt_ldattach = FALSE;
	bool opt_lddetach = FALSE;
	bool opt_namespace = FALSE;
	bool opt_configure = FALSE;
	bool opt_unconfigure = FALSE;
	bool opt_service = FALSE;
	bool opt_unservice = FALSE;
	bool opt_libbasedir = FALSE;
	bool opt_patch = FALSE;
	bool opt_starthost = FALSE;
	bool opt_stophost = FALSE;
	bool opt_passthrough = FALSE;

	char *arg_ldattach;
	char *arg_configure;
	char *arg_service;
	char *arg_namespace;
	char *arg_libbasedir;
	char *arg_patch;

	int index;
    for (;;) {
		index = 0;
        int opt = getopt_long(argc, argv, "+:uh:a:d:n:l:f:p:c:s:rz", opts, &index);
        if (opt == -1) {
            break;
        }
        switch (opt) {
		case 'a':
			opt_ldattach = TRUE;
			arg_ldattach = optarg;
			break;
		case 'd':
			opt_lddetach = TRUE;
			arg_lddetach = optarg;
			break;
		case 'n':
			opt_namespace = TRUE;
			arg_namespace = optarg;
			break;
		case 'c':
			opt_configure = TRUE;
			arg_configure = optarg;
			break;
		case 'w':
			opt_unconfigure = TRUE;
			break;
		case 's':
			opt_service = TRUE;
			arg_service = optarg;
			break;
		case 'v':
			opt_unservice = TRUE;
			break;
		case 'f':
		case 'l':
			opt_libbasedir = TRUE;
			arg_libbasedir = optarg;
			break;
		case 'p':
			opt_patch = TRUE;
			arg_patch = optarg;
			break;
		case 'r':
			opt_starthost = TRUE;
			break;
		case 'x':
			opt_stophost = TRUE;
			break;
		case 'z':
			opt_passthrough = TRUE;
			break;
		case ':': // Handle options missing their arg value
			switch (optopt) {
			default:
				fprintf(stderr, "error: missing required value for -%c option\n", optopt);
				return EXIT_FAILURE;
			}
			break;
		default:
			break;
        }
    }

	// Handle potential argument conflicts
	if (opt_ldattach && opt_lddetach) {
        fprintf(stderr, "error: --ldattach and --lddetach cannot be used together\n");
        return EXIT_FAILURE;
	}
	if (opt_service && opt_unservice) {
        fprintf(stderr, "error: --service and --unservice cannot be used together\n");
        return EXIT_FAILURE;
	}
	if (opt_configure && opt_unconfigure) {
        fprintf(stderr, "error: --configure/--unconfigure cannot be used together\n");
        return EXIT_FAILURE;
	}
    if ((opt_ldattach || opt_lddetach) && (opt_service || opt_unservice)) {
        fprintf(stderr, "error: --ldattach/--lddetach and --service/--unservice cannot be used together\n");
        return EXIT_FAILURE;
    }
    if ((opt_ldattach || opt_lddetach) && (opt_configure || opt_unconfigure)) {
        fprintf(stderr, "error: --ldattach/--lddetach and --configure/--unconfigure cannot be used together\n");
        return EXIT_FAILURE;
    }
    if ((opt_configure || unconfigure) && (opt_service || opt_unservice)) {
        fprintf(stderr, "error: --configure/--unconfigure and --service/--unservice cannot be used together\n");
        return EXIT_FAILURE;
    }
    if (opt_namespace && (!opt_configure && !opt_unconfigure && !opt_service && !opt_unservice)) {
        fprintf(stderr, "error: --namespace option requires --configure/--unconfigure or --service/--unservice option\n");
        return EXIT_FAILURE;
    }
	if (opt_passthrough) && (opt_ldattach || opt_lddetach || opt_namespace ||
		opt_service || opt_unservice || opt_configure || opt_unconfigure) {
        fprintf(stderr, "error: --passthrough cannot be used with --ldattach/--lddetach or --namespace or --service/--unservice or --configure/--unconfigure\n");
        return EXIT_FAILURE;
	}

	// Process options for use with commands
	if (opt_libbasedir) {
		if (libdirSetLibraryBase(arg_libbasedir)) {
			return EXIT_FAILURE;
		}
	}

	// Execute commands
	if (opt_ldattach) cmdRun(opt_ldattach, FALSE); // Will exit
	if (opt_lddetach) cmdRun(FALSE, opt_lddetach); // Will exit
	if (opt_configure) exit(cmdConfigure());
	if (opt_unconfigure) exit(cmdUnconfigure());
	if (opt_service) exit(cmdService());
	if (opt_unservice) exit(cmdUnservice());
	if (opt_patch) exit(loaderOpPatchLibrary(optarg) == PATCH_SUCCESS);
	if (opt_starthost) return nsHostStart();
	if (opt_stophost) return nsHostStop();
	if (opt_passthrough) cmdRun(FALSE, FALSE); // Will exit

	// No constructor command executed.
	// Continue to regular CLI usage.
	return;
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

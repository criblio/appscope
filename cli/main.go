package main

/*
#if defined(__aarch64__)
#cgo LDFLAGS: -L../lib/linux/aarch64 -lloader
#else
#cgo LDFLAGS: -L../lib/linux/x86_64 -lloader
#endif
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <unistd.h>
#include <limits.h>
#include <string.h>
#include <fcntl.h>

#include "../src/loader/libdir.h"
#include "../src/loader/loader.h"
#include "../src/loader/patch.h"
#include "../src/loader/ns.h"

// Example Usage:
// scope [OPTIONS] --ldattach PID
// scope [OPTIONS] --lddetach PID
// scope [OPTIONS] --rules RULES_PATH --rootdir /hostfs
// scope [OPTIONS] --service SERVICE --namespace PID
// scope [OPTIONS] --passthrough EXECUTABLE [ARGS...]
// scope [OPTIONS] --patch SO_FILE

// Options:
// -l, --libbasedir DIR              specify parent for the library directory (default: /tmp)
// -a, --ldattach PID                attach to the specified process ID
// -d, --lddetach PID                detach from the specified process ID
// -i, --install                     install libscope.so and scope
// -e, --preload PATH                set ld.so.preload to PATH. "auto" = auto detect libpath; "off" = disable
// -f, --rules RULES_PATH            install the rules file specified in RULES_PATH
// -m, --mount MOUNT_DEST            mount rules file and unix socket into MOUNT_DEST
// -R, --rootdir PATH                specify root directory of the target namespace
// -s, --service SERVICE             setup specified service NAME
// -v, --unservice                   remove scope from all service configurations
// -n  --namespace PID               perform operation on specified container PID
// -p, --patch SO_FILE               patch specified libscope.so
// -z, --passthrough                 scope a command, bypassing all cli set up

// Long aliases for short options
// NOTE: Be sure to align these with the options listed in the call to getopt_long
// NOTE: These must not conflict with the cli options specified in the cmd/ package
static struct option opts[] = {
	{ "ldattach",    required_argument, 0, 'a' },
	{ "lddetach",    required_argument, 0, 'd' },
	{ "preload",     required_argument, 0, 'e' },
	{ "rules",       required_argument, 0, 'f' },
	{ "mount",       required_argument, 0, 'm' },
	{ "install",     no_argument,       0, 'i' },
	{ "libbasedir",  required_argument, 0, 'l' },
	{ "namespace",   required_argument, 0, 'n' },
	{ "patch",       required_argument, 0, 'p' },
	{ "rootdir",     required_argument, 0, 'R' },
	{ "service",     required_argument, 0, 's' },
	{ "unservice",   no_argument,       0, 'v' },
	{ "passthrough", no_argument,       0, 'z' },
	{ 0, 0, 0, 0 }
};

unsigned long g_libscopesz;
unsigned long g_scopedynsz;

// This is the constructor for the Go application.
// Code here executes before the Go Runtime starts.
// We execute loader-specific commands here, because we can perform namespace switches while meeting
// the OS requirement of being in a single-threaded process to do so.
// Note: We cannot rely on argc/argv/environ being present,
// i.e. the musl library does not define them as constructor args.
__attribute__((constructor)) void cli_constructor() {
	FILE *f;
	int c;
	char fname[PATH_MAX];
	int scope_pid = getpid();
	int arg_max = 1024; // Is this reasonable?
	int arg_c = 0;
	size_t size = 0;
	char **arg_v;
	int index;
	pid_t nspid = -1;
	pid_t pid = -1;
	uid_t eUid = geteuid();
	int cmdArgc;
	char **cmdArgv;

	bool opt_ldattach = false;
	bool opt_lddetach = false;
	bool opt_namespace = false;
	bool opt_install = false;
	bool opt_preload = false;
	bool opt_rules = false;
	bool opt_mount = false;
	bool opt_rootdir = false;
	bool opt_service = false;
	bool opt_unservice = false;
	bool opt_libbasedir = false;
	bool opt_patch = false;
	bool opt_passthrough = false;

	char *arg_ldattach = NULL;
	char *arg_lddetach = NULL;
	char *arg_rootdir = NULL;
	char *arg_preload = NULL;
	char *arg_rules = NULL;
	char *arg_mount = NULL;
	char *arg_service = NULL;
	char *arg_namespace = NULL;
	char *arg_libbasedir = NULL;
	char *arg_patch = NULL;

	if ((g_libscopesz = strtoul(LIBSCOPE_SO_SIZE, NULL, 10)) == ULONG_MAX) {
	   perror("strtoul");
	   exit(EXIT_FAILURE);
	}

	if ((g_scopedynsz = strtoul(SCOPEDYN_SIZE, NULL, 10)) == ULONG_MAX) {
	   perror("strtoul");
	   exit(EXIT_FAILURE);
	}

	// Open /proc/pid/cmdline to get args
	snprintf(fname, sizeof fname, "/proc/%d/cmdline", scope_pid);
	f = fopen(fname, "r");
	if (!f) {
		perror("fopen");
		exit(EXIT_FAILURE);
	}

	// Read args delimited by null
	arg_v = malloc(sizeof(char *) * arg_max);
	if (!arg_v) {
		perror("malloc");
		exit(EXIT_FAILURE);
	}
	while(getdelim(&arg_v[arg_c], &size, 0, f) != -1)
	{
		arg_c++;
	}
	arg_v[arg_c] = NULL;

	fclose(f);

	for (;;) {
		index = 0;
		int opt = getopt_long(arg_c, arg_v, "+:a:d:n:m:l:f:p:e:s:R:vzi", opts, &index);
		if (opt == -1) {
			break;
		}
		switch (opt) {
		case 'a':
			opt_ldattach = true;
			arg_ldattach = optarg;
			break;
		case 'd':
			opt_lddetach = true;
			arg_lddetach = optarg;
			break;
		case 'i':
			opt_install = true;
			break;
		case 'R':
			opt_rootdir = true;
			arg_rootdir = optarg;
			break;
		case 'n':
			opt_namespace = true;
			arg_namespace = optarg;
			break;
		case 's':
			opt_service = true;
			arg_service = optarg;
			break;
		case 'v':
			opt_unservice = true;
			break;
		case 'e':
			opt_preload = true;
			arg_preload = optarg;
			break;
		case 'f':
			opt_rules = true;
			arg_rules = optarg;
			break;
		case 'm':
			opt_mount = true;
			arg_mount = optarg;
			break;
		case 'l':
			opt_libbasedir = true;
			arg_libbasedir = optarg;
			break;
		case 'p':
			opt_patch = true;
			arg_patch = optarg;
			break;
		case 'z':
			opt_passthrough = true;
			break;
		case ':': // Handle options missing their arg value
			switch (optopt) {
			default:
				fprintf(stderr, "error: missing required value for -%c option\n", optopt);
				exit(EXIT_FAILURE);
			}
			break;
		default:
			break;
		}
	}

	// Handle potential argument conflicts
	// TODO this is getting too long, let's make a function that covers all the bases with easier syntax
	if (opt_install && (opt_ldattach || opt_lddetach)) {
		fprintf(stderr, "error: --install and --ldattach/lddetach cannot be used together\n");
		exit(EXIT_FAILURE);
	}
	if (opt_ldattach && opt_lddetach) {
		fprintf(stderr, "error: --ldattach and --lddetach cannot be used together\n");
		exit(EXIT_FAILURE);
	}
	if (opt_service && opt_unservice) {
		fprintf(stderr, "error: --service and --unservice cannot be used together\n");
		exit(EXIT_FAILURE);
	}
	if ((opt_ldattach || opt_lddetach) && (opt_service || opt_unservice)) {
		fprintf(stderr, "error: --ldattach/--lddetach and --service/--unservice cannot be used together\n");
		exit(EXIT_FAILURE);
	}
	if ((opt_ldattach || opt_lddetach) && (opt_rules)) {
		fprintf(stderr, "error: --ldattach/--lddetach and --rules cannot be used together\n");
		exit(EXIT_FAILURE);
	}
	if ((opt_rules) && (opt_service || opt_unservice)) {
		fprintf(stderr, "error: --rules and --service/--unservice cannot be used together\n");
		exit(EXIT_FAILURE);
	}
	if (opt_namespace && (!opt_service && !opt_unservice)) {
		fprintf(stderr, "error: --namespace option requires --service/--unservice option\n");
		exit(EXIT_FAILURE);
	}
	if (opt_passthrough && (opt_ldattach || opt_lddetach || opt_namespace ||
		opt_service || opt_unservice || opt_rules || opt_preload)) {
		fprintf(stderr, "error: --passthrough cannot be used with --ldattach/--lddetach or --namespace or --service/--unservice or --rules or --preload\n");
		exit(EXIT_FAILURE);
	}

	// Handle potential permissions issues
	if (eUid && (opt_rootdir || opt_service || opt_unservice)) {
		fprintf(stderr, "error: command requires root\n");
		exit(EXIT_FAILURE);
	}

	// Process options for use with commands
	if (opt_libbasedir) {
		if (libdirSetLibraryBase(arg_libbasedir)) {
			exit(EXIT_FAILURE);
		}
	}
	if (opt_namespace) {
		nspid = atoi(arg_namespace);
		if (nspid < 1) {
			fprintf(stderr, "error: invalid --namespace PID: %s\n", arg_namespace);
			exit(EXIT_FAILURE);
		}
	}
	if (opt_ldattach) {
		pid = atoi(arg_ldattach);
		if (pid < 1) {
			fprintf(stderr, "error: invalid --ldattach PID: %s\n", arg_ldattach);
			exit(EXIT_FAILURE);
		}
	}
	if (opt_lddetach) {
		pid = atoi(arg_lddetach);
		if (pid < 1) {
			fprintf(stderr, "error: invalid --lddetach PID: %s\n", arg_lddetach);
			exit(EXIT_FAILURE);
		}
	}

	// Execute commands
	cmdArgc = arg_c-optind; // argc of the program we want to scope
	cmdArgv = &arg_v[optind]; // argv of the program we want to scope

	if (opt_ldattach) exit(cmdAttach(pid, arg_rootdir));
	if (opt_lddetach) exit(cmdDetach(pid, arg_rootdir));
	if (opt_install) exit(cmdInstall(arg_rootdir));
	if (opt_rules) exit(cmdRules(arg_rules, arg_rootdir));
	if (opt_preload) exit(cmdPreload(arg_preload, arg_rootdir));
	if (opt_mount) exit(cmdMount(arg_mount, arg_rootdir));
	if (opt_service) exit(cmdService(arg_service, nspid));
	if (opt_unservice) exit(cmdUnservice(nspid));
	if (opt_patch) exit(patchLibrary(arg_patch, FALSE) == PATCH_FAILED);
	if (opt_passthrough) exit(cmdRun(pid, nspid, cmdArgc, cmdArgv));

	// No constructor command executed.
	// Continue to regular CLI usage.
	if (arg_v) free(arg_v);
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

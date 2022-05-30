#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <syslog.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/wait.h>
#include <errno.h>
#include <string.h>
#include <elf.h>
#include <libgen.h>
#include <dirent.h>
#include <getopt.h>
#include <sys/utsname.h>

#include "scopestdlib.h"
#include "scopetypes.h"
#include "libdir.h"

/*
 * This code exists solely to support the ability to
 * build a libscope and an ldscope on a glibc distro
 * that will execute on both a glibc distro and a
 * musl distro.
 *
 * This code is used to create a static exec that will
 * execute on both glibc and musl distros.
 *
 * The process:
 * 1) extract the ldscopedyn dynamic exec and libscope.so
 *    dynamic lib from this object.
 * 2) open an executable file on the current FS and
 *    read the loader string from the .interp section.
 * 3) if it uses a musl ld.so then do musl
 * 4) for musl; create a dir and in that dir create a
 *    soft link to ld-musl.so from ld.linux.so (the
 *    glibc loader).
 * 5) for musl; create or add to the ld lib path
 *    env var to point to the dir created above.
 * 6) for musl; modify the loader string in .interp
 *    of ldscope to ld-musl.so.
 * 7) execve the extracted ldscopedyn passing args
 *    from this command line.
 */

#define EXE_TEST_FILE "/bin/cat"
#define LIBMUSL "musl"

static int g_debug = 0;

static int
get_dir(const char *path, char *fres, size_t len)
{
    DIR *dirp;
    struct dirent *entry;
    char *dcopy, *pcopy, *dname, *fname;
    int res = -1;

    if (!path || !fres || (len <= 0)) return res;

    pcopy = scope_strdup(path);
    dname = scope_dirname(pcopy);

    if ((dirp = scope_opendir(dname)) == NULL) {
        scope_perror("get_dir:opendir");
        if (pcopy) scope_free(pcopy);
        return res;
    }

    dcopy = scope_strdup(path);
    fname = scope_basename(dcopy);

    while ((entry = scope_readdir(dirp)) != NULL) {
        if ((entry->d_type != DT_DIR) &&
            (scope_strstr(entry->d_name, fname))) {
            scope_strncpy(fres, entry->d_name, len);
            res = 0;
            break;
        }
    }

    scope_closedir(dirp);
    if (pcopy) scope_free(pcopy);
    if (dcopy) scope_free(dcopy);
    return res;
}

static void
setEnvVariable(char *env, char *value)
{
    char *cur_val = getenv(env);

    // If env is not set
    if (!cur_val) {
        if (setenv(env, value, 1)) {
            perror("setEnvVariable:setenv");
        }
        return;
    }

    // env is set. try to append
    char *new_val = NULL;
    if ((scope_asprintf(&new_val, "%s:%s", cur_val, value) == -1)) {
        scope_perror("setEnvVariable:asprintf");
        return;
    }

    if (g_debug) scope_printf("%s:%d %s to %s\n", __FUNCTION__, __LINE__, env, new_val);
    if (setenv(env, new_val, 1)) {
        perror("setEnvVariable:setenv");
    }

    if (new_val) scope_free(new_val);
}

// modify NEEDED entries in libscope.so to avoid dependencies
static int
set_library(const char *libpath)
{
    int i, fd, found, name;
    struct stat sbuf;
    char *buf;
    Elf64_Ehdr *elf;
    Elf64_Shdr *sections;
    Elf64_Dyn *dyn;
    const char *section_strtab = NULL;
    const char *strtab = NULL;
    const char *sec_name = NULL;

    if (libpath == NULL)
        return -1;

    if ((fd = scope_open(libpath, O_RDONLY)) == -1) {
        scope_perror("set_library:open");
        return -1;
    }

    if (scope_fstat(fd, &sbuf) == -1) {
        scope_perror("set_library:fstat");
        scope_close(fd);
        return -1;
    }

    buf = scope_mmap(NULL, ROUND_UP(sbuf.st_size, scope_sysconf(_SC_PAGESIZE)),
               PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, (off_t)NULL);
    if (buf == MAP_FAILED) {
        scope_perror("set_loader:scope_mmap");
        scope_close(fd);
        return -1;
    }

    // get the elf header, section table and string table
    elf = (Elf64_Ehdr *)buf;

    if (elf->e_ident[EI_MAG0] != ELFMAG0
        || elf->e_ident[EI_MAG1] != ELFMAG1
        || elf->e_ident[EI_MAG2] != ELFMAG2
        || elf->e_ident[EI_MAG3] != ELFMAG3
        || elf->e_ident[EI_VERSION] != EV_CURRENT) {
        scope_fprintf(scope_stderr, "ERROR:%s: is not valid ELF file", libpath);
        scope_close(fd);
        scope_munmap(buf, sbuf.st_size);
        return -1;
    }

    sections = (Elf64_Shdr *)((char *)buf + elf->e_shoff);
    section_strtab = (char *)buf + sections[elf->e_shstrndx].sh_offset;
    found = name = 0;

    // locate the .dynstr section
    for (i = 0; i < elf->e_shnum; i++) {
        sec_name = section_strtab + sections[i].sh_name;
        if (sections[i].sh_type == SHT_STRTAB && scope_strcmp(sec_name, ".dynstr") == 0) {
            strtab = (const char *)(buf + sections[i].sh_offset);
        }
    }

    if (strtab == NULL) {
        scope_fprintf(scope_stderr, "ERROR:%s: did not locate the .dynstr from %s", __FUNCTION__, libpath);
        scope_close(fd);
        scope_munmap(buf, sbuf.st_size);
        return -1;
    }

    // locate the .dynamic section
    for (i = 0; i < elf->e_shnum; i++) {
        if (sections[i].sh_type == SHT_DYNAMIC) {
            for (dyn = (Elf64_Dyn *)((char *)buf + sections[i].sh_offset); dyn != NULL && dyn->d_tag != DT_NULL; dyn++) {
                if (dyn->d_tag == DT_NEEDED) {
                    char *depstr = (char *)(strtab + dyn->d_un.d_val);
                    if (depstr && scope_strstr(depstr, "ld-linux")) {
                        char newdep[PATH_MAX];
                        size_t newdep_len;
                        if (get_dir("/lib/ld-musl", newdep, sizeof(newdep)) == -1) break;
                        newdep_len = scope_strlen(newdep);
                        if (scope_strlen(depstr) >= newdep_len) {
                            scope_strncpy(depstr, newdep, newdep_len + 1);
                            found = 1;
                            break;
                        }
                    }
                }
            }
        }
        if (found == 1) break;
    }

    if (found) {
        if (scope_close(fd) == -1) {
            scope_munmap(buf, sbuf.st_size);
            return -1;
        }

        if ((fd = scope_open(libpath, O_RDWR)) == -1) {
            scope_perror("set_library:open write");
            scope_munmap(buf, sbuf.st_size);
            return -1;
        }

        int rc = scope_write(fd, buf, sbuf.st_size);
        if (rc < sbuf.st_size) {
            scope_perror("set_library:write");
        }
    }

    scope_close(fd);
    scope_munmap(buf, sbuf.st_size);
    return (found - 1);
}

// modify the loader string in the .interp section of ldscope
static int
set_loader(char *exe)
{
    int i, fd, found, name;
    struct stat sbuf;
    char *buf;
    Elf64_Ehdr *elf;
    Elf64_Phdr *phead;

    if (!exe) return -1;

    if ((fd = scope_open(exe, O_RDONLY)) == -1) {
        scope_perror("set_loader:open");
        return -1;
    }

    if (scope_fstat(fd, &sbuf) == -1) {
        scope_perror("set_loader:fstat");
        scope_close(fd);
        return -1;
    }

    buf = scope_mmap(NULL, ROUND_UP(sbuf.st_size, scope_sysconf(_SC_PAGESIZE)),
               PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, (off_t)NULL);
    if (buf == MAP_FAILED) {
        scope_perror("set_loader:scope_mmap");
        scope_close(fd);
        return -1;
    }

    elf = (Elf64_Ehdr *)buf;
    phead  = (Elf64_Phdr *)&buf[elf->e_phoff];
    found = name = 0;

    for (i = 0; i < elf->e_phnum; i++) {
        if ((phead[i].p_type == PT_INTERP)) {
            char *exld = (char *)&buf[phead[i].p_offset];
            DIR *dirp;
            struct dirent *entry;
            char dir[PATH_MAX];
            size_t dir_len;

            if (scope_strstr(exld, "ld-musl") != NULL) {
                scope_close(fd);
                scope_munmap(buf, sbuf.st_size);
                return 0;
            }

            scope_snprintf(dir, sizeof(dir), "/lib/");
            if ((dirp = scope_opendir(dir)) == NULL) {
                scope_perror("set_loader:opendir");
                break;
            }

            while ((entry = scope_readdir(dirp)) != NULL) {
                if ((entry->d_type != DT_DIR) &&
                    (scope_strstr(entry->d_name, "ld-musl"))) {
                    scope_strncat(dir, entry->d_name, scope_strlen(entry->d_name) + 1);
                    name = 1;
                    break;
                }
            }

            scope_closedir(dirp);
            dir_len = scope_strlen(dir);
            if (name && (scope_strlen(exld) >= dir_len)) {
                if (g_debug) scope_printf("%s:%d exe ld.so: %s to %s\n", __FUNCTION__, __LINE__, exld, dir);
                scope_strncpy(exld, dir, dir_len + 1);
                found = 1;
                break;
            }
        }
    }

    if (found) {
        if (scope_close(fd) == -1) {
            scope_munmap(buf, sbuf.st_size);
            return -1;
        }

        if ((fd = scope_open(exe, O_RDWR)) == -1) {
            scope_perror("set_loader:open write");
            scope_munmap(buf, sbuf.st_size);
            return -1;
        }

        int rc = scope_write(fd, buf, sbuf.st_size);
        if (rc < sbuf.st_size) {
            scope_perror("set_loader:write");
        }
    } else {
        scope_fprintf(scope_stderr, "WARNING: can't locate or set the loader string in %s\n", exe);
    }

    scope_close(fd);
    scope_munmap(buf, sbuf.st_size);
    return (found - 1);
}

static char *
get_loader(char *exe)
{
    int i, fd;
    struct stat sbuf;
    char *buf, *ldso = NULL;
    Elf64_Ehdr *elf;
    Elf64_Phdr *phead;

    if (!exe) return NULL;

    if ((fd = scope_open(exe, O_RDONLY)) == -1) {
        scope_perror("get_loader:open");
        return NULL;
    }

    if (scope_fstat(fd, &sbuf) == -1) {
        scope_perror("get_loader:fstat");
        scope_close(fd);
        return NULL;
    }

    buf = scope_mmap(NULL, ROUND_UP(sbuf.st_size, scope_sysconf(_SC_PAGESIZE)),
               PROT_READ, MAP_PRIVATE, fd, (off_t)NULL);
    if (buf == MAP_FAILED) {
        scope_perror("get_loader:scope_mmap");
        scope_close(fd);
        return NULL;
    }

    scope_close(fd);

    elf = (Elf64_Ehdr *)buf;
    phead  = (Elf64_Phdr *)&buf[elf->e_phoff];

    for (i = 0; i < elf->e_phnum; i++) {
        if ((phead[i].p_type == PT_INTERP)) {
            char * exld = (char *)&buf[phead[i].p_offset];
            if (g_debug) scope_printf("%s:%d exe ld.so: %s\n", __FUNCTION__, __LINE__, exld);

            ldso = scope_strdup(exld);

            break;
        }
    }

    scope_munmap(buf, sbuf.st_size);
    return ldso;
}

static void
do_musl(char *exld, char *ldscope)
{
    char *lpath = NULL;
    char *ldso = NULL;
    char *path;
    char dir[scope_strlen(ldscope) + 2];

    // always set the env var
    scope_strncpy(dir, ldscope, scope_strlen(ldscope) + 1);
    path = scope_dirname(dir);
    setEnvVariable(LD_LIB_ENV, path);

    // does a link to the musl ld.so exist?
    // if so, we assume the symlink exists as well.
    if ((ldso = get_loader(ldscope)) == NULL) return;

    // Avoid creating ld-musl-x86_64.so.1 -> /lib/ld-musl-x86_64.so.1
    if (scope_strstr(ldso, "musl")) return;

    if (scope_asprintf(&lpath, "%s/%s", path, basename(ldso)) == -1) {
        scope_perror("do_musl:asprintf");
        if (ldso) scope_free(ldso);
        return;
    }

    // dir is expected to exist here, not creating one
    if ((symlink((const char *)exld, lpath) == -1) &&
        (errno != EEXIST)) {
        scope_perror("do_musl:symlink");
        if (ldso) scope_free(ldso);
        if (lpath) scope_free(lpath);
        return;
    }

    set_loader(ldscope);
    set_library(libdirGetLibrary());

    if (ldso) scope_free(ldso);
    if (lpath) scope_free(lpath);
}

/*
 * Check for and handle the extra steps needed to run under musl libc.
 *
 * Returns 0 if musl was not detected and 1 if it was.
 */
static int
setup_loader(char *ldscope)
{
    int ret = 0; // not musl

    char *ldso = NULL;

    if (((ldso = get_loader(EXE_TEST_FILE)) != NULL) &&
        (scope_strstr(ldso, LIBMUSL) != NULL)) {
            // we are using the musl ld.so
            do_musl(ldso, ldscope);
            ret = 1; // detected musl
    }

    if (ldso) scope_free(ldso);

    return ret;
}

static int
patch_library(const char *so_path) {
    int result = EXIT_FAILURE;

    char *ldso = get_loader(EXE_TEST_FILE);
    if (ldso && scope_strstr(ldso, LIBMUSL) != NULL) {
        result = set_library(optarg);
    }
    scope_free(ldso);

    return result;
}

/* 
 * This avoids a segfault when code using shm_open() is compiled statically.
 * For some reason, compiling the code statically causes the __shm_directory()
 * function calls in librt.a to not reach the implementation in libpthread.a.
 * Implementing the function ourselves fixes this issue.
 *
 * See https://stackoverflow.com/a/47914897
 */
#ifndef  SHM_MOUNT
#define  SHM_MOUNT "/dev/shm/"
#endif
static const char  shm_mount[] = SHM_MOUNT;
const char *__shm_directory(size_t *len)
{
    if (len)
        *len = scope_strlen(shm_mount);
    return shm_mount;
}

static const char scope_help_overview[] =
"  OVERVIEW:\n"
"    The Scope library supports extraction of data from within applications.\n"
"    As a general rule, applications consist of one or more processes.\n"
"    The Scope library can be loaded into any process as the\n"
"    process starts.\n"
"    The primary way to define which processes include the Scope library\n"
"    is by exporting the environment variable LD_PRELOAD, which is set to point\n"
"    to the path name of the Scope library. E.g.: \n"
"    export LD_PRELOAD=./libscope.so\n"
"\n"
"    Scope emits data as metrics and/or events.\n"
"    Scope is fully configurable by means of a configuration file (scope.yml)\n"
"    and/or environment variables.\n"
"\n"
"    Metrics are emitted in StatsD format, over a configurable link. By default,\n"
"    metrics are sent over a UDP socket using localhost and port 8125.\n"
"\n"
"    Events are emitted in JSON format over a configurable link. By default,\n"
"    events are sent over a TCP socket using localhost and port 9109.\n"
"\n"
"    Scope logs to a configurable destination, at a configurable\n"
"    verbosity level. The default verbosity setting is level 4, and the\n"
"    default destination is the file `/tmp/scope.log`.\n";

static const char scope_help_configuration[] =
"  CONFIGURATION:\n"
"    Configuration File:\n"
"        A YAML config file (named scope.yml) enables control of all available\n"
"        settings. The config file is optional. Environment variables take\n"
"        precedence over settings in a config file.\n"
"\n"
"    Config File Resolution\n"
"        If the SCOPE_CONF_PATH env variable is defined, and points to a\n"
"        file that can be opened, it will use this as the config file.\n"
"        Otherwise, AppScope searches for the config file in this priority\n"
"        order, using the first one it finds:\n"
"\n"
"            $SCOPE_HOME/conf/scope.yml\n"
"            $SCOPE_HOME/scope.yml\n"
"            /etc/scope/scope.yml\n"
"            ~/conf/scope.yml\n"
"            ~/scope.yml\n"
"            ./conf/scope.yml\n"
"            ./scope.yml\n"
"\n"
"    Environment Variables:\n"
"        SCOPE_CONF_PATH\n"
"            Directly specify config file's location and name.\n"
"            Used only at start time.\n"
"        SCOPE_HOME\n"
"            Specify a directory from which conf/scope.yml or ./scope.yml can\n"
"            be found. Used only at start time, and only if SCOPE_CONF_PATH\n"
"            does not exist. For more info, see Config File Resolution below.\n"
"        SCOPE_METRIC_ENABLE\n"
"            Single flag to make it possible to disable all metric output.\n"
"            true,false  Default is true.\n"
"        SCOPE_METRIC_VERBOSITY\n"
"            0-9 are valid values. Default is 4.\n"
"            For more info, see Metric Verbosity below.\n"
"        SCOPE_METRIC_FS\n"
"            Create metrics describing file connectivity.\n"
"            true, false  Default is true.\n"
"        SCOPE_METRIC_NET\n"
"            Create metrics describing network connectivity.\n"
"            true, false  Default is true.\n"
"        SCOPE_METRIC_HTTP\n"
"            Create metrics describing HTTP communication.\n"
"            true, false  Default is true.\n"
"        SCOPE_METRIC_DNS\n"
"            Create metrics describing DNS activity.\n"
"            true, false  Default is true.\n"
"        SCOPE_METRIC_PROC\n"
"            Create metrics describing process.\n"
"            true, false  Default is true.\n"
"        SCOPE_METRIC_STATSD\n"
"            When true, statsd metrics sent or received by this application\n"
"            will be handled as appscope-native metrics.\n"
"            true, false  Default is true.\n"
"        SCOPE_METRIC_DEST\n"
"            Default is udp://localhost:8125\n"
"            Format is one of:\n"
"                file:///tmp/output.log\n"
"                    Output to a file. Use file://stdout, file://stderr for\n"
"                    STDOUT or STDERR\n"
"                udp://host:port\n"
"                tcp://host:port\n"
"                    Send to a TCP or UDP server. \"host\" is the hostname or\n"
"                    IP address and \"port\" is the port number of service name.\n"
"                unix://socketpath\n"
"                    Output to a unix domain server using TCP.\n"
"                    Use unix://@abstractname, unix:///var/run/mysock for\n"
"                    abstract address or filesystem address.\n"
"        SCOPE_METRIC_TLS_ENABLE\n"
"            Flag to enable Transport Layer Security (TLS). Only affects\n"
"            tcp:// destinations. true,false  Default is false.\n"
"        SCOPE_METRIC_TLS_VALIDATE_SERVER\n"
"            false allows insecure (untrusted) TLS connections, true uses\n"
"            certificate validation to ensure the server is trusted.\n"
"            Default is true.\n"
"        SCOPE_METRIC_TLS_CA_CERT_PATH\n"
"            Path on the local filesystem which contains CA certificates in\n"
"            PEM format. Default is an empty string. For a description of what\n"
"            this means, see Certificate Authority Resolution below.\n"
"        SCOPE_METRIC_FORMAT\n"
"            statsd, ndjson\n"
"            Default is statsd.\n"
"        SCOPE_STATSD_PREFIX\n"
"            Specify a string to be prepended to every scope metric.\n"
"        SCOPE_STATSD_MAXLEN\n"
"            Default is 512.\n"
"        SCOPE_SUMMARY_PERIOD\n"
"            Number of seconds between output summarizations. Default is 10.\n"
"        SCOPE_EVENT_ENABLE\n"
"            Single flag to make it possible to disable all event output.\n"
"            true,false  Default is true.\n"
"        SCOPE_EVENT_DEST\n"
"            Same format as SCOPE_METRIC_DEST above.\n"
"            Default is tcp://localhost:9109\n"
"        SCOPE_EVENT_TLS_ENABLE\n"
"            Flag to enable Transport Layer Security (TLS). Only affects\n"
"            tcp:// destinations. true,false  Default is false.\n"
"        SCOPE_EVENT_TLS_VALIDATE_SERVER\n"
"            false allows insecure (untrusted) TLS connections, true uses\n"
"            certificate validation to ensure the server is trusted.\n"
"            Default is true.\n"
"        SCOPE_EVENT_TLS_CA_CERT_PATH\n"
"            Path on the local filesystem which contains CA certificates in\n"
"            PEM format. Default is an empty string. For a description of what\n"
"            this means, see Certificate Authority Resolution below.\n"
"        SCOPE_EVENT_FORMAT\n"
"            ndjson\n"
"            Default is ndjson.\n"
"        SCOPE_EVENT_LOGFILE\n"
"            Create events from writes to log files.\n"
"            true,false  Default is false.\n"
"        SCOPE_EVENT_LOGFILE_NAME\n"
"            An extended regex to filter log file events by file name.\n"
"            Used only if SCOPE_EVENT_LOGFILE is true. Default is .*log.*\n"
"        SCOPE_EVENT_LOGFILE_VALUE\n"
"            An extended regex to filter log file events by field value.\n"
"            Used only if SCOPE_EVENT_LOGFILE is true. Default is .*\n"
"        SCOPE_EVENT_CONSOLE\n"
"            Create events from writes to stdout, stderr.\n"
"            true,false  Default is false.\n"
"        SCOPE_EVENT_CONSOLE_NAME\n"
"            An extended regex to filter console events by event name.\n"
"            Used only if SCOPE_EVENT_CONSOLE is true. Default is .*\n"
"        SCOPE_EVENT_CONSOLE_VALUE\n"
"            An extended regex to filter console events by field value.\n"
"            Used only if SCOPE_EVENT_CONSOLE is true. Default is .*\n"
"        SCOPE_EVENT_METRIC\n"
"            Create events from metrics.\n"
"            true,false  Default is false.\n"
"        SCOPE_EVENT_METRIC_NAME\n"
"            An extended regex to filter metric events by event name.\n"
"            Used only if SCOPE_EVENT_METRIC is true. Default is .*\n"
"        SCOPE_EVENT_METRIC_FIELD\n"
"            An extended regex to filter metric events by field name.\n"
"            Used only if SCOPE_EVENT_METRIC is true. Default is .*\n"
"        SCOPE_EVENT_METRIC_VALUE\n"
"            An extended regex to filter metric events by field value.\n"
"            Used only if SCOPE_EVENT_METRIC is true. Default is .*\n"
"        SCOPE_EVENT_HTTP\n"
"            Create events from HTTP headers.\n"
"            true,false  Default is false.\n"
"        SCOPE_EVENT_HTTP_NAME\n"
"            An extended regex to filter http events by event name.\n"
"            Used only if SCOPE_EVENT_HTTP is true. Default is .*\n"
"        SCOPE_EVENT_HTTP_FIELD\n"
"            An extended regex to filter http events by field name.\n"
"            Used only if SCOPE_EVENT_HTTP is true. Default is .*\n"
"        SCOPE_EVENT_HTTP_VALUE\n"
"            An extended regex to filter http events by field value.\n"
"            Used only if SCOPE_EVENT_HTTP is true. Default is .*\n"
"        SCOPE_EVENT_HTTP_HEADER\n"
"            An extended regex that defines user defined headers\n"
"            that will be extracted. Default is NULL\n"
"        SCOPE_EVENT_NET\n"
"            Create events describing network connectivity.\n"
"            true,false  Default is false.\n"
"        SCOPE_EVENT_NET_NAME\n"
"            An extended regex to filter network events by event name.\n"
"            Used only if SCOPE_EVENT_NET is true. Default is .*\n"
"        SCOPE_EVENT_NET_FIELD\n"
"            An extended regex to filter network events by field name.\n"
"            Used only if SCOPE_EVENT_NET is true. Default is .*\n"
"        SCOPE_EVENT_NET_VALUE\n"
"            An extended regex to filter network events by field value.\n"
"            Used only if SCOPE_EVENT_NET is true. Default is .*\n"
"        SCOPE_EVENT_FS\n"
"            Create events describing file connectivity.\n"
"            true,false  Default is false.\n"
"        SCOPE_EVENT_FS_NAME\n"
"            An extended regex to filter file events by event name.\n"
"            Used only if SCOPE_EVENT_FS is true. Default is .*\n"
"        SCOPE_EVENT_FS_FIELD\n"
"            An extended regex to filter file events by field name.\n"
"            Used only if SCOPE_EVENT_FS is true. Default is .*\n"
"        SCOPE_EVENT_FS_VALUE\n"
"            An extended regex to filter file events by field value.\n"
"            Used only if SCOPE_EVENT_FS is true. Default is .*\n"
"        SCOPE_EVENT_DNS\n"
"            Create events describing DNS activity.\n"
"            true,false  Default is false.\n"
"        SCOPE_EVENT_DNS_NAME\n"
"            An extended regex to filter dns events by event name.\n"
"            Used only if SCOPE_EVENT_DNS is true. Default is .*\n"
"        SCOPE_EVENT_DNS_FIELD\n"
"            An extended regex to filter DNS events by field name.\n"
"            Used only if SCOPE_EVENT_DNS is true. Default is .*\n"
"        SCOPE_EVENT_DNS_VALUE\n"
"            An extended regex to filter dns events by field value.\n"
"            Used only if SCOPE_EVENT_DNS is true. Default is .*\n"
"        SCOPE_EVENT_MAXEPS\n"
"            Limits number of events that can be sent in a single second.\n"
"            0 is 'no limit'; 10000 is the default.\n"
"        SCOPE_ENHANCE_FS\n"
"            Controls whether uid, gid, and mode are captured for each open.\n"
"            Used only if SCOPE_EVENT_FS is true. true,false Default is true.\n"
"        SCOPE_LOG_LEVEL\n"
"            debug, info, warning, error, none. Default is error.\n"
"        SCOPE_LOG_DEST\n"
"            same format as SCOPE_METRIC_DEST above.\n"
"            Default is file:///tmp/scope.log\n"
"        SCOPE_LOG_TLS_ENABLE\n"
"            Flag to enable Transport Layer Security (TLS). Only affects\n"
"            tcp:// destinations. true,false  Default is false.\n"
"        SCOPE_LOG_TLS_VALIDATE_SERVER\n"
"            false allows insecure (untrusted) TLS connections, true uses\n"
"            certificate validation to ensure the server is trusted.\n"
"            Default is true.\n"
"        SCOPE_LOG_TLS_CA_CERT_PATH\n"
"            Path on the local filesystem which contains CA certificates in\n"
"            PEM format. Default is an empty string. For a description of what\n"
"            this means, see Certificate Authority Resolution below.\n"
"        SCOPE_TAG_\n"
"            Specify a tag to be applied to every metric and event.\n"
"            Environment variable expansion is available, \n"
"            e.g.: SCOPE_TAG_user=$USER\n"
"        SCOPE_CMD_DIR\n"
"            Specifies a directory to look for dynamic configuration files.\n"
"            See Dynamic Configuration below.\n"
"            Default is /tmp\n"
"        SCOPE_PAYLOAD_ENABLE\n"
"            Flag that enables payload capture.  true,false  Default is false.\n"
"        SCOPE_PAYLOAD_DIR\n"
"            Specifies a directory where payload capture files can be written.\n"
"            Default is /tmp\n"
"        SCOPE_CRIBL_ENABLE\n"
"            Single flag to make it possible to disable cribl backend.\n"
"            true,false  Default is true.\n"
"        SCOPE_CRIBL_CLOUD\n"
"            Intended as an alternative to SCOPE_CRIBL below. Identical\n"
"            behavior, except that where SCOPE_CRIBL can have TLS settings\n"
"            modified via related SCOPE_CRIBL_TLS_* environment variables,\n"
"            SCOPE_CRIBL_CLOUD hardcodes TLS settings as though these were\n"
"            individually specified:\n"
"                SCOPE_CRIBL_TLS_ENABLE=true\n"
"                SCOPE_CRIBL_TLS_VALIDATE_SERVER=true\n"
"                SCOPE_CRIBL_TLS_CA_CERT_PATH=\"\"\n"
"            As a note, library behavior will be undefined if this variable is\n"
"            set simultaneously with SCOPE_CRIBL, or any of SCOPE_CRIBL_TLS_*.\n"
"        SCOPE_CRIBL\n"
"            Defines a connection with Cribl LogStream\n"
"            Default is NULL\n"
"            Format is one of:\n"
"                tcp://host:port\n"
"                    If no port is provided, defaults to 10090\n"
"                unix://socketpath\n"
"                    Output to a unix domain server using TCP.\n"
"                    Use unix://@abstractname, unix:///var/run/mysock for\n"
"                    abstract address or filesystem address.\n"
"        SCOPE_CRIBL_AUTHTOKEN\n"
"            Authentication token provided by Cribl.\n"
"            Default is an empty string.\n"
"        SCOPE_CRIBL_TLS_ENABLE\n"
"            Flag to enable Transport Layer Security (TLS). Only affects\n"
"            tcp:// destinations. true,false  Default is false.\n"
"        SCOPE_CRIBL_TLS_VALIDATE_SERVER\n"
"            false allows insecure (untrusted) TLS connections, true uses\n"
"            certificate validation to ensure the server is trusted.\n"
"            Default is true.\n"
"        SCOPE_CRIBL_TLS_CA_CERT_PATH\n"
"            Path on the local filesystem which contains CA certificates in\n"
"            PEM format. Default is an empty string. For a description of what\n"
"            this means, see Certificate Authority Resolution below.\n"
"        SCOPE_CONFIG_EVENT\n"
"            Sends a single process-identifying event, when a transport\n"
"            connection is established.  true,false  Default is true.\n"
"\n"
"    Dynamic Configuration:\n"
"        Dynamic Configuration allows configuration settings to be\n"
"        changed on the fly after process start time. At every\n"
"        SCOPE_SUMMARY_PERIOD, the library looks in SCOPE_CMD_DIR to\n"
"        see if a file scope.<pid> exists. If it exists, the library processes\n"
"        every line, looking for environment variable–style commands\n"
"        (e.g., SCOPE_CMD_DBG_PATH=/tmp/outfile.txt). The library changes the\n"
"        configuration to match the new settings, and deletes the\n"
"        scope.<pid> file when it's complete.\n"
"\n"
"    Certificate Authority Resolution\n"
"        If SCOPE_*_TLS_ENABLE and SCOPE_*_TLS_VALIDATE_SERVER are true then\n"
"        AppScope performs TLS server validation. For this to be successful\n"
"        a CA certificate must be found that can authenticate the certificate\n"
"        the server provides during the TLS handshake process.\n"
"        If SCOPE_*_TLS_CA_CERT_PATH is set, AppScope will use this file path\n"
"        which is expected to contain CA certificates in PEM format. If this\n"
"        env variable is an empty string or not set, AppScope searches for a\n"
"        usable root CA file on the local filesystem, using the first one\n"
"        found from this list:\n"
"\n"
"            /etc/ssl/certs/ca-certificates.crt\n"
"            /etc/pki/tls/certs/ca-bundle.crt\n"
"            /etc/ssl/ca-bundle.pem\n"
"            /etc/pki/tls/cacert.pem\n"
"            /etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem\n"
"            /etc/ssl/cert.pem\n";

static const char scope_help_metrics[] =
"  METRICS:\n"
"    Metrics can be enabled or disabled with a single config element\n"
"    (metric: enable: true|false). Specific types of metrics, and specific \n"
"    field content, are managed with a Metric Verbosity setting.\n"
"\n"
"    Metric Verbosity\n"
"        Controls two different aspects of metric output – \n"
"        Tag Cardinality and Summarization.\n"
"\n"
"        Tag Cardinality\n"
"            0   No expanded StatsD tags\n"
"            1   adds 'data', 'unit'\n"
"            2   adds 'class', 'proto'\n"
"            3   adds 'op'\n"
"            4   adds 'pid', 'host', 'proc', 'http_status'\n"
"            5   adds 'domain', 'file'\n"
"            6   adds 'localip', 'remoteip', 'localp', 'port', 'remotep'\n"
"            7   adds 'fd', 'args'\n"
"            8   adds 'duration','numops','req_per_sec','req','resp','protocol'\n"
"\n"
"        Summarization\n"
"            0-4 has full event summarization\n"
"            5   turns off 'error'\n"
"            6   turns off 'filesystem open/close' and 'dns'\n"
"            7   turns off 'filesystem stat' and 'network connect'\n"
"            8   turns off 'filesystem seek'\n"
"            9   turns off 'filesystem read/write' and 'network send/receive'\n"
"\n"
"    The http.status metric is emitted when the http watch type has been\n"
"    enabled as an event. The http.status metric is not controlled with\n"
"    summarization settings.\n";

static const char scope_help_events[] =
"  EVENTS:\n"
"    All events can be enabled or disabled with a single config element\n"
"    (event: enable: true|false). Unlike metrics, event content is not \n"
"    managed with verbosity settings. Instead, you use regex filters that \n"
"    manage which field types and field values to include.\n"
"\n"
"     Events are organized as 7 watch types: \n"
"     1) File Content. Provide a pathname, and all data written to the file\n"
"        will be organized in JSON format and emitted over the event channel.\n"
"     2) Console Output. Select stdin and/or stdout, and all data written to\n"
"        these endpoints will be formatted in JSON and emitted over the event\n"
"        channel.\n"
"     3) Metrics. Event metrics provide the greatest level of detail from\n"
"        libscope. Events are created for every read, write, send, receive,\n"
"        open, close, and connect. These raw events are configured with regex\n"
"        filters to manage which event, which specific fields within an event,\n"
"        and which value patterns within a field to include.\n"
"     4) HTTP Headers. HTTP headers are extracted, formatted in JSON, and\n"
"        emitted over the event channel. Three types of events are created\n"
"        for HTTP headers: 1) HTTP request events, 2) HTTP response events,\n"
"        and 3) a metric event corresponding to the request and response\n"
"        sequence. A response event includes the corresponding request,\n"
"        status and duration fields. An HTTP metric event provides fields\n"
"        describing bytes received, requests per second, duration, and status.\n"
"        Any header defined as X-appscope (case insensitive) will be emitted.\n"
"        User defined headers are extracted by using the headers field.\n"
"        The headers field is a regular expression.\n"
"     5) File System. Events are formatted in JSON for each file system open,\n"
"        including file name, permissions, and cgroup. Events for file system\n"
"        close add a summary of the number of bytes read and written, the\n"
"        total number of read and write operations, and the total duration\n"
"        of read and write operations. The specific function performing open\n"
"        and close is reported as well.\n"
"     6) Network. Events are formatted in JSON for network connections and \n"
"        corresponding disconnects, including type of protocol used, and \n"
"        local and peer IP:port. Events for network disconnect add a summary\n"
"        of the number of bytes sent and received, and the duration of the\n"
"        sends and receives while the connection was active. The reason\n"
"        (source) for disconnect is provided as local or remote. \n"
"     7) DNS. Events are formatted in JSON for DNS requests and responses.\n"
"        The event provides the domain name being resolved. On DNS response,\n"
"        the event provides the duration of the DNS operation.\n";

static const char scope_help_protocol[] =
"  PROTOCOL DETECTION:\n"
"    Scope can detect any defined network protocol. You provide protocol\n"
"    definitions in the \"protocol\" section of the config file. You describe \n"
"    protocol specifics in one or more regex definitions. PCRE2 regular \n"
"    expressions are supported. The stock scope.yml file for examples.\n"
"\n"
"    Scope detects binary and string protocols. Detection events, \n"
"    formatted in JSON, are emitted over the event channel unless the \"detect\"\n"
"    property is set to \"false\".\n"
"\n"
"  PAYLOAD EXTRACTION:\n"
"    When enabled, libscope extracts payload data from network operations.\n"
"    Payloads are emitted in binary. No formatting is applied to the data.\n"
"    Payloads are emitted to either a local file or the LogStream channel.\n"
"    Configuration elements for libscope support defining a path for payload\n"
"    data.\n";

static int
showHelp(const char *section)
{
    scope_printf(
      "Cribl AppScope Static Loader %s\n"
      "\n"
      "AppScope is a general-purpose observable application telemetry system.\n"
      "\n",
      SCOPE_VER
    );

    if (!section || !scope_strcasecmp(section, "all")) {
        scope_puts(scope_help_overview);
        scope_puts(scope_help_configuration);
        scope_puts(scope_help_metrics);
        scope_puts(scope_help_events);
        scope_puts(scope_help_protocol);
    } else if (!scope_strcasecmp(section, "overview")) {
        scope_puts(scope_help_overview);
    } else if (!scope_strcasecmp(section, "configuration") || !scope_strcasecmp(section, "config")) {
        scope_puts(scope_help_configuration);
    } else if (!scope_strcasecmp(section, "metrics")) {
        scope_puts(scope_help_metrics);
    } else if (!scope_strcasecmp(section, "events")) {
        scope_puts(scope_help_events);
    } else if (!scope_strcasecmp(section, "protocols")) {
        scope_puts(scope_help_protocol);
    } else {
        scope_fprintf(scope_stderr, "error: invalid help section\n\n");
        return -1;
    }
    scope_fflush(scope_stdout);
    return 0;
}

static void
showUsage(char *prog)
{
    scope_printf(
      "\n"
      "Cribl AppScope Static Loader %s\n" 
      "\n"
      "AppScope is a general-purpose observable application telemetry system.\n"
      "\n"
      "usage: %s [OPTIONS] [--] EXECUTABLE [ARGS...]\n"
      "       %s [OPTIONS] --attach PID\n"
      "\n"
      "options:\n"
      "  -u, --usage           display this info\n"
      "  -h, --help [SECTION]  display all or the specified help section\n"
      "  -l, --libbasedir DIR  specify parent for the library directory (default: /tmp)\n"
      "  -f DIR                alias for \"-l DIR\" for backward compatibility\n"
      "  -a, --attach PID      attach to the specified process ID\n"
      "  -p, --patch SO_FILE   patch specified libscope.so \n"
      "\n"
      "Help sections are OVERVIEW, CONFIGURATION, METRICS, EVENTS, and PROTOCOLS.\n"
      "\n"
      "See `scope` if you are new to AppScope as it provides a simpler and more\n"
      "user-friendly experience as you come up to speed.\n"
      "\n"
      "User docs are at https://appscope.dev/docs/. The project is hosted at\n"
      "https://github.com/criblio/appscope. Please direct feature requests and\n"
      "defect reports there.\n"
      "\n",
      SCOPE_VER, prog, prog
    );
    scope_fflush(scope_stdout);
}

// long aliases for short options
static struct option opts[] = {
    { "usage",      no_argument,       0, 'u'},
    { "help",       optional_argument, 0, 'h' },
    { "attach",     required_argument, 0, 'a' },
    { "libbasedir", required_argument, 0, 'l' },
    { "patch",      required_argument, 0, 'p' },
    { 0, 0, 0, 0 }
};

int
main(int argc, char **argv, char **env)
{
    char *attachArg = 0;
    char path[PATH_MAX] = {0};

    // process command line
    for (;;) {
        int index = 0;
        //
        // The `+` here enables POSIX mode where the first non-option found
        // stops option processing so `ldscope foo -a 123` will not process the
        // `-a 123` here and instead pass it through.
        //
        // The initial `:` lets us handle options with optional values like
        // `-h` and `-h SECTION`.
        //
        int opt = getopt_long(argc, argv, "+:uh:a:l:f:p:", opts, &index);
        if (opt == -1) {
            break;
        }
        switch (opt) {
            case 'u':
                showUsage(scope_basename(argv[0]));
                return EXIT_SUCCESS;
            case 'h':
                // handle `-h SECTION`
                if (showHelp(optarg)) {
                    showUsage(scope_basename(argv[0]));
                    return EXIT_FAILURE;
                }
                return EXIT_SUCCESS;
            case 'a':
                attachArg = optarg;
                break;
            case 'f':
                // accept -f as alias for -l for BC
            case 'l':
                libdirSetBase(optarg);
                break;
            case 'p':
                return patch_library(optarg);
            case ':':
                // options missing their value end up here
                switch (optopt) {
                    case 'h':
                        // handle `-h` without the section value
                        showHelp(0);
                        return EXIT_SUCCESS;
                    default: 
                        scope_fprintf(scope_stderr, "error: missing required value for -%c option\n", optopt);
                        showUsage(scope_basename(argv[0]));
                        return EXIT_FAILURE;
                }
                break;
            case '?':
            default:
                scope_fprintf(scope_stderr, "error: invalid option: -%c\n", optopt);
                showUsage(scope_basename(argv[0]));
                return EXIT_FAILURE;
        }
    }

    // either --attach or a command are required
    if (!attachArg && optind >= argc) {
        scope_fprintf(scope_stderr, "error: missing --attach option or EXECUTABLE argument\n");
        showUsage(scope_basename(argv[0]));
        return EXIT_FAILURE;
    }

    // use --attach, ignore executable and args
    if (attachArg && optind < argc) {
        scope_fprintf(scope_stderr, "warning: ignoring EXECUTABLE argument with --attach option\n");
    }

    // extract to the library directory
    if (libdirExtractLoader()) {
        scope_fprintf(scope_stderr, "error: failed to extract loader\n");
        return EXIT_FAILURE;
    }

    if (libdirExtractLibrary()) {
        scope_fprintf(scope_stderr, "error: failed to extract library\n");
        return EXIT_FAILURE;
    }

    // setup for musl libc if detected
    char *loader = (char *)libdirGetLoader();
    if (!loader) {
        scope_fprintf(scope_stderr, "error: failed to get a loader path\n");
        return EXIT_FAILURE;
    }
    setup_loader(loader);

    // set SCOPE_EXEC_PATH to path to `ldscope` if not set already
    if (getenv("SCOPE_EXEC_PATH") == 0) {
        char execPath[PATH_MAX];
        if (scope_readlink("/proc/self/exe", execPath, sizeof(execPath) - 1) == -1) {
            scope_perror("readlink(/proc/self/exe) failed");
            return EXIT_FAILURE;
        }
        setenv("SCOPE_EXEC_PATH", execPath, 0);
    }

    // create /dev/shm/scope_${PID}.env when attaching
    if (attachArg) {
        // must be root
        if (scope_getuid()) {
            scope_printf("error: --attach requires root\n");
            return EXIT_FAILURE;
        }

        // target process must exist
        int pid = scope_atoi(attachArg);
        if (pid < 1) {
            scope_printf("error: invalid --attach PID: %s\n", attachArg);
            return EXIT_FAILURE;
        }
        scope_snprintf(path, sizeof(path), "/proc/%d", pid);
        if (scope_access(path, F_OK)) {
            scope_printf("error: --attach PID not a current process: %d\n", pid);
            return EXIT_FAILURE;
        }

        // create .env file for the library to load
        scope_snprintf(path, sizeof(path), "/scope_attach_%d.env", pid);
        int fd = scope_shm_open(path, O_RDWR|O_CREAT, S_IRUSR|S_IRGRP|S_IROTH);
        if (fd == -1) {
            scope_perror("shm_open() failed");
            return EXIT_FAILURE;
        }

        // add the env vars we want in the library
        scope_dprintf(fd, "SCOPE_LIB_PATH=%s\n", libdirGetLibrary());

        int i;
        for (i = 0; environ[i]; i++) {
            if (scope_strlen(environ[i]) > 6 && scope_strncmp(environ[i], "SCOPE_", 6) == 0) {
                scope_dprintf(fd, "%s\n", environ[i]);
            }
        }

        // done
        scope_close(fd);
    }

    // build exec args
    int execArgc = 0;
    char **execArgv = scope_calloc(argc + 4, sizeof(char *));
    if (!execArgv) {
        scope_perror("scope_calloc");
        return EXIT_FAILURE;
    }

    execArgv[execArgc++] = (char *) libdirGetLoader();

    if (attachArg) {
        execArgv[execArgc++] = "-a";
        execArgv[execArgc++] = attachArg;
    } else {
        while (optind < argc) {
            execArgv[execArgc++] = argv[optind++];
        }
    }

    execArgv[execArgc++] = NULL;

    // pass SCOPE_LIB_PATH in environment
    if (setenv("SCOPE_LIB_PATH", libdirGetLibrary(), 1)) {
        scope_perror("setenv(SCOPE_LIB_PATH) failed");
        return EXIT_FAILURE;
    }

    // exec the dynamic ldscope
    struct utsname ubuf;

    if (scope_uname(&ubuf) != 0) {
        scope_perror("uname");
        return EXIT_FAILURE;
    }

    execve(libdirGetLoader(), execArgv, environ);

    scope_free(execArgv);
    scope_perror("execve failed");
    return EXIT_FAILURE;
}

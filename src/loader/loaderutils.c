#define _GNU_SOURCE
#include <dlfcn.h>
#include <fcntl.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <limits.h>
#include <sys/types.h>
#include <dirent.h>

#include "scopetypes.h"
#include "loader.h"
#include "loaderutils.h"

void
freeElf(char *buf, size_t len)
{
    if (!buf) return;

    if (munmap(buf, len) == -1) {
        fprintf(stderr, "freeElf: munmap failed\n");
    }
}

static void
setTextSizeAndLenFromElf(elf_buf_t *ebuf)
{
    int i;
    Elf64_Ehdr *ehdr;
    Elf64_Shdr *sections;
    const char *sec_name;
    const char *section_strtab = NULL;
    ehdr = (Elf64_Ehdr *)ebuf->buf;
    sections = (Elf64_Shdr *)((char *)ebuf->buf + ehdr->e_shoff);
    section_strtab = (char *)ebuf->buf + sections[ehdr->e_shstrndx].sh_offset;

    for (i = 0; i < ehdr->e_shnum; i++) {
        sec_name = section_strtab + sections[i].sh_name;

        if (!strcmp(sec_name, ".text")) {
            ebuf->text_addr = (unsigned char *)sections[i].sh_addr;
            ebuf->text_len = sections[i].sh_size;
            if (g_log_level <= CFG_LOG_DEBUG) {
                fprintf(stderr, "%s:%d %s addr %p - %p\n", __FUNCTION__, __LINE__,
                       sec_name, ebuf->text_addr, ebuf->text_addr + ebuf->text_len);
            }
        }
    }
}

static bool
app_type(char *buf, const uint32_t sh_type, const char *sh_name)
{
    int i = 0;
    Elf64_Ehdr *ehdr = (Elf64_Ehdr *)buf;
    Elf64_Shdr *sections;
    const char *section_strtab = NULL;
    const char *sec_name = NULL;

    sections = (Elf64_Shdr *)(buf + ehdr->e_shoff);
    section_strtab = buf + sections[ehdr->e_shstrndx].sh_offset;

    for (i = 0; i < ehdr->e_shnum; i++) {
        sec_name = section_strtab + sections[i].sh_name;
        //printf("section %s type = %d \n", sec_name, sections[i].sh_type);
        if (sections[i].sh_type == sh_type && strcmp(sec_name, sh_name) == 0) {
            return TRUE;
        }
    }
    return FALSE;
}

elf_buf_t *
getElf(char *path)
{
    int fd = -1;
    elf_buf_t *ebuf = NULL;
    Elf64_Ehdr *elf;
    struct stat sbuf;
    int get_elf_successful = FALSE;


    if ((ebuf = calloc(1, sizeof(elf_buf_t))) == NULL) {
        fprintf(stderr, "getElf: memory alloc failed");
        goto out;
    }

    if ((fd = open(path, O_RDONLY)) == -1) {
        fprintf(stderr, "getElf: open failed");
        goto out;
    }

    if (fstat(fd, &sbuf) == -1) {
        fprintf(stderr, "fd:%d getElf: fstat failed", fd);
        goto out;
    }


    char * mmap_rv = mmap(NULL, ROUND_UP(sbuf.st_size, sysconf(_SC_PAGESIZE)),
                          PROT_READ, MAP_PRIVATE, fd, (off_t)NULL);
    if (mmap_rv == MAP_FAILED) {
        fprintf(stderr, "fd:%d getElf: mmap failed", fd);
        goto out;
    }

    ebuf->cmd = path;
    ebuf->buf = mmap_rv;
    ebuf->len = sbuf.st_size;

    elf = (Elf64_Ehdr *)ebuf->buf;
    if((elf->e_ident[EI_MAG0] != 0x7f) ||
       strncmp((char *)&elf->e_ident[EI_MAG1], "ELF", 3) ||
       (elf->e_ident[EI_CLASS] != ELFCLASS64) ||
       (elf->e_ident[EI_DATA] != ELFDATA2LSB)) {
        fprintf(stderr, "fd:%d %s:%d ERROR: %s is not a viable ELF file\n",
                fd, __FUNCTION__, __LINE__, path);
        goto out;
    }

    if ((elf->e_type != ET_EXEC) && (elf->e_type != ET_DYN)) {
        fprintf(stderr, "fd:%d %s:%d %s with type %d is not an executable\n",
                fd, __FUNCTION__, __LINE__, path, elf->e_type);
        goto out;
    }

    setTextSizeAndLenFromElf(ebuf);

    get_elf_successful = TRUE;

out:
    if (fd != -1) close(fd);
    if (!get_elf_successful && ebuf) {
        freeElf(ebuf->buf, ebuf->len);
        free(ebuf);
        ebuf = NULL;
    }
    return ebuf;
}

bool
is_static(char *buf)
{
    int i;
    Elf64_Ehdr *elf = (Elf64_Ehdr *)buf;
    Elf64_Phdr *phead = (Elf64_Phdr *)&buf[elf->e_phoff];

    for (i = 0; i < elf->e_phnum; i++) {
        if ((phead[i].p_type == PT_DYNAMIC) || (phead[i].p_type == PT_INTERP)) {
            return FALSE;
        }
    }

    return TRUE;
}

bool
is_go(char *buf)
{
    if (buf && (app_type(buf, SHT_PROGBITS, ".gosymtab") ||
                app_type(buf, SHT_PROGBITS, ".gopclntab") ||
                app_type(buf, SHT_NOTE, ".note.go.buildid"))) {
        return TRUE;
    }

    return FALSE;
}

void
setPidEnv(int pid)
{
    char val[32];
    int returnval = snprintf(val, sizeof(val), "%d", pid);
    if (returnval >= sizeof(val) || returnval == -1) {
        return;
    }

    if ((setenv(SCOPE_PID_ENV, val, 1) == -1) &&
        (g_log_level <= CFG_LOG_DEBUG)) {
        fprintf(stderr, "setPidEnv: %s:%s", SCOPE_PID_ENV, val);
    }
}

// This tests to see if cmd can be resolved to an executable file.
// If so it will return a malloc'd buffer containing an absolute path,
// otherwise it will return NULL.
char *
getpath(const char *cmd)
{
    char *path_env = NULL;
    char *ret_val = NULL;
    struct stat buf;

    if (!cmd) goto out;

    // an absolute path was specified for cmd.
    if (cmd[0] == '/') {
        //  If we can resolve it, use it.
        if (!stat(cmd, &buf) && S_ISREG(buf.st_mode) && (buf.st_mode & 0111)) {
            ret_val = strdup(cmd);
        }
        goto out;
    }

    // a relative path was specified for cmd.
    if (strchr(cmd, '/')) {
        char *cur_dir = get_current_dir_name();
        if (!cur_dir) goto out;

        char *path = NULL;
        if (asprintf(&path, "%s/%s", cur_dir, cmd) > 0) {
            // If we can resolve it, use it
            if (!stat(path, &buf) && S_ISREG(buf.st_mode) && (buf.st_mode & 0111)) {
                ret_val = path;
            } else {
                free(path);
            }
        }
        free(cur_dir);
        goto out;
    }

    // try the current dir
    char *path = NULL;
    if (asprintf(&path, "./%s", cmd) > 0) {
        if (!stat(path, &buf) && S_ISREG(buf.st_mode) && (buf.st_mode & 0111)) {
            ret_val = path;
            goto out;
        } else {
            free(path);
        }
    }

    // try to resolve the cmd from PATH env variable
    char *path_env_ptr = getenv("PATH");
    if (!path_env_ptr) goto out;
    path_env = strdup(path_env_ptr); // create a copy for strtok below
    if (!path_env) goto out;

    char *saveptr = NULL;
    char *strtok_path = strtok_r(path_env, ":", &saveptr);
    if (!strtok_path) goto out;

    do {
        char *path = NULL;
        if (asprintf(&path, "%s/%s", strtok_path, cmd) < 0) {
            break;
        }
        if ((stat(path, &buf) == -1) ||    // path doesn't exist
            (!S_ISREG(buf.st_mode)) ||     // path isn't a file
            ((buf.st_mode & 0111) == 0)) { // path isn't executable

            free(path);
            continue;
        }

        // We found the cmd, and it's an executable file
        ret_val = path;
        break;

    } while ((strtok_path = strtok_r(NULL, ":", &saveptr)));

out:
    if (path_env) free(path_env);
    return ret_val;
}

uint64_t
findLibrary(const char *library, pid_t pid, bool matchLibraryExactly)
{
    char filename[PATH_MAX];
    char line[1024];
    FILE *fd;
    uint64_t addr = 0;

    snprintf(filename, sizeof(filename), "/proc/%d/maps", pid);
    if ((fd = fopen(filename, "r")) == NULL) {
        // return no proc found as opposed to no libscope found
        return -1;
    }

    if (matchLibraryExactly) { // pathFromLine string == library string
        char pathFromLine[512];
        while(fgets(line, sizeof(line), fd)) {
            //7ff775e88000-7ff77606f000 r-xp 00000000 ca:01 17348 /lib/x86_64-linux-gnu/libc-2.27.so
            if ((sscanf(line, "%*x-%*x %*s %*x %*s %*d %512s", pathFromLine) == 1)
                && !strcmp(pathFromLine, library)) {
                addr = strtoull(line, NULL, 16);
                break;
            }
        }
    } else {                   // line *contains* library string
        while(fgets(line, sizeof(line), fd)) {
            if (strstr(line, library)) {
                addr = strtoull(line, NULL, 16);
                break;
            }
        }
    }

    fclose(fd);
    return addr;
}

int
getExePath(pid_t pid, char **path)
{
    if (!path) return -1;
    char *buf = *path;
    char pidpath[PATH_MAX];

    if (!(buf = calloc(1, PATH_MAX))) {
        fprintf(stderr, "ERROR:calloc in osGetExePath");
        return -1;
    }

    snprintf(pidpath, PATH_MAX, "/proc/%d/exe", pid);

    if (readlink(pidpath, buf, PATH_MAX - 1) == -1) {
        fprintf(stderr, "osGetExePath: can't get path to pid %d exe", pid);
        free(buf);
        return -1;
    }

    *path = buf;
    return 0;
}

int
findFd(pid_t pid, const char *fname)
{
    int fd = -1;
    DIR *dirp;
    char *cwd;
    struct dirent *entry;
    char buf[PATH_MAX], link[256];

    if ((cwd = getcwd(NULL, 0)) == NULL) {
        perror("getcwd");
        return -1;
    }

    snprintf(buf, sizeof(buf), "/proc/%d/fd", pid);

    if (chdir(buf) == -1) {
        free(cwd);
        return -1;
    }

    if ((dirp = opendir(buf)) == NULL) {
        free(cwd);
        perror("opendir");
        return -1;
    }

    while ((entry = readdir(dirp)) != NULL) {
        if (entry->d_type != DT_DIR) {
                if (readlink(entry->d_name, link, sizeof(link)) == -1) {
                    fprintf(stderr, "%s: can't get path to %s", __FUNCTION__, entry->d_name);
                break;
            }

            if (strstr(link, fname)) {
                fd = strtol(entry->d_name, NULL, 0);
                if ((fd == LONG_MIN) || (fd == LONG_MAX) || (fd == 0)) fd = -1;
                break;
            }
        }
    }

    chdir(cwd);
    closedir(dirp);
    free(cwd);
    return fd;
}

int
getProcUidGid(pid_t pid, uid_t *uid, gid_t *gid)
{
    char path[PATH_MAX] = {0};
    char buffer[4096];
    uid_t eUid = -1;
    gid_t eGid = -1;

    if (snprintf(path, sizeof(path), "/proc/%d/status", pid) < 0) return -1;

    FILE *fstream = fopen(path, "r");
    if (fstream == NULL) {
        return -1;
    }

    while (fgets(buffer, sizeof(buffer), fstream)) {
        const char delimiters[] = ": \t";
        if (strstr(buffer, "Uid:")) {
            char *entry, *last;
            // Skip Uid string
            entry = strtok_r(buffer, delimiters, &last);
            // Get real Uid value
            entry = strtok_r(NULL, delimiters, &last);
            eUid = atoi(entry);
        }

        if (strstr(buffer, "Gid:")) {
            char *entry, *last;
            // Skip Gid string
            entry = strtok_r(buffer, delimiters, &last);
            // Get real Gid value
            entry = strtok_r(NULL, delimiters, &last);
            eGid = atoi(entry);
        }
    }

    fclose(fstream);

    if (eUid != -1 && eGid != -1) {
        *uid = eUid;
        *gid = eGid;
        return 0;
    }

    return -1;
}


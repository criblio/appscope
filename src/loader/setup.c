#define _GNU_SOURCE

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include "libdir.h"
#include "libver.h"
#include "nsfile.h"
#include "patch.h"
#include "scopetypes.h"
#include "setup.h"

#define BUFSIZE (4096)

#define OPENRC_DIR "/etc/rc.conf"
#define SYSTEMD_DIR "/etc/systemd"
#define INITD_DIR "/etc/init.d"
#define PROFILE_SCRIPT "#! /bin/bash\nlib_found=0\nfilter_found=0\nldpreload=`printenv LD_PRELOAD`\nif test -f /usr/lib/appscope/%s/libscope.so; then\n    lib_found=1\nfi\nif test -f /usr/lib/appscope/scope_filter; then\n    filter_found=1\nelif test -f /tmp/appscope/scope_filter; then\n    filter_found=1\nfi\nif [ $lib_found == 1 ] && [ $filter_found == 1 ]; then\n    if [ -n \"$ldpreload\" ]; then\n        if [[ $ldpreload == *\"libscope.so\"* ]]; then\n            if [[ $ldpreload == *\":\"* ]]; then\n                delim=\":\"\n                libscope=`echo  $ldpreload | awk -F':' '{for(i=1;i<=NF;i++) if($i~\"libscope.so\") print substr($i, 1, length($i))}'`\n            else\n                delim=\" \"\n                libscope=`echo $ldpreload | awk -F' ' '{for(i=1;i<=NF;i++) if($i~\"libscope.so\") print substr($i, 1, length($i))}'`\n            fi\n            nolibscope=${ldpreload//$libscope}\n            export LD_PRELOAD=\"%s$delim$nolibscope\"\n        fi\n    else\n        export LD_PRELOAD=\"%s\"\n    fi\nfi\n"

typedef enum {
    SERVICE_CFG_ERROR,
    SERVICE_CFG_NEW,
    SERVICE_CFG_EXIST,
} service_cfg_status_t;

/*
 * Helper function to check if specified service is already configured to preload scope.
 * Returns TRUE if service is already configured FALSE otherwise.
 */
bool
isCfgFileConfigured(const char *serviceCfgPath) {
    FILE *fPtr;
    int res = FALSE;
    char buf[BUFSIZE] = {0};

    if ((fPtr = fopen(serviceCfgPath, "r")) == NULL) {
        perror("isCfgFileConfigured fopen failed");
        return res;
    }

    while(fgets(buf, sizeof(buf), fPtr)) {
        // TODO improve it to verify particular version ?
        if (strstr(buf, "/libscope.so")) {
            res = TRUE;
            break;
        }
    }

    fclose(fPtr);

    return res;
}

/*
 * Helper function to remove any lines containing "/libscope.so" from service configs
 * Reads and Writes characters one at a time to avoid the requirement for a fixed size buffer
 * Returns number of lines modified ; 0 if no file was modified; -1 on error
 */
int
removeScopeCfgFile(const char *filePath) {
    FILE *f1, *f2;
    int c;
    char *tempPath = "/tmp/tmpFile-XXXXXX";
    bool newline = TRUE;
    char line_buf[128];
    int count = 0;
     
    f1 = fopen(filePath, "r");
    if (!f1) {
        return -1;
    }
    f2 = fopen(tempPath, "w");
    if (!f2) {
        fclose(f1);
        return -1;
    }

    while ((c = getc(f1)) != EOF) {
        long file_pos = ftell(f1); // Save file position 
        if (c == '\n') {
            putc(c, f2);
            newline = TRUE;
            continue;
        }
        if (newline) {
            fseek(f1, file_pos - 1, SEEK_SET); // Rewind file position to beginning of line
            fgets(line_buf, sizeof(line_buf), f1);
            if (strstr(line_buf, "/libscope.so")) {
                // Skip over this line, effectively removing it from the new file
                count++;
                newline = TRUE;
                continue;
            }
            fseek(f1, file_pos, SEEK_SET); // Rewind file position previous point
        }
        putc(c, f2);
        newline = FALSE;
    }

    fclose(f1);
    fclose(f2);

    fprintf(stderr, "info: Modifying service file %s\n", filePath);

    if (remove(filePath)) {
        fprintf(stderr, "error: Removing original service file %s\n", filePath);
        return -1;
    }
    if (rename(tempPath, filePath)) {
        fprintf(stderr, "error: Moving newly created service file %s\n", filePath);
        return -1;
    }

    return count;
}

/*
 * Check if specified service is installed in Systemd service manager.
 *
 * Returns TRUE if service is installed FALSE otherwise.
 */
static bool
isServiceInstalledSystemD(const char *serviceName) {

    /*
    * List of directories which can contain service configuration file.
    */
    const char *const servicePrefixList[] = {
        "/etc/systemd/system",
        "/lib/systemd/system",
        "/run/systemd/system",
        "/usr/lib/systemd/system"
    };

    for (int i = 0; i < sizeof(servicePrefixList)/sizeof(char*); ++i) {
        char cfgPath[PATH_MAX] = {0};
        struct stat st = {0};
        if (snprintf(cfgPath, sizeof(cfgPath), "%s/%s.service", servicePrefixList[i], serviceName) < 0) {
            perror("error: isServiceInstalledSystemD, snprintf failed");
            return FALSE;
        }

        if (stat(cfgPath, &st) == 0) {
            return TRUE;
        }
    }
    return FALSE;
}

/*
 * Check if specified service is installed in InitD/OpenRc service manager.
 *
 * Returns TRUE if service is installed FALSE otherwise.
 */
static bool
isServiceInstalledInitDOpenRc(const char *serviceName) {
    char cfgPath[PATH_MAX] = {0};
    struct stat st = {0};
    if (snprintf(cfgPath, sizeof(cfgPath), "/etc/init.d/%s", serviceName) < 0) {
        perror("error: isServiceInstalledInitDOpenRc, snprintf failed");
        return FALSE;
    }

    return (stat(cfgPath, &st) == 0) ? TRUE : FALSE;
}

/*
 * Get Systemd service configuration file status.
 *
 * Returns status of the Service file.
 */
static service_cfg_status_t
serviceCfgStatusSystemD(const char *serviceName, uid_t uid, gid_t gid) {
    char cfgScript[PATH_MAX] = {0};
    struct stat st = {0};
    service_cfg_status_t ret = SERVICE_CFG_ERROR;

    if (snprintf(cfgScript, sizeof(cfgScript), "/etc/systemd/system/%s.service.d/", serviceName) < 0) {
        perror("error: serviceCfgStatusSystemD, snprintf failed");
        return ret;
    }

    // create service.d directory if it does not exists.
    if (stat(cfgScript, &st) != 0) {
        if (nsFileMkdir(cfgScript, 0755, uid, gid, geteuid(), getegid()) != 0) {
            perror("error: serviceCfgStatusSystemD, mkdir failed");
            return ret;
        }
    }

    strncat(cfgScript, "env.conf", C_STRLEN("env.conf") + 1);

    if (stat(cfgScript, &st) == 0) {
        ret = SERVICE_CFG_EXIST;
    } else {
        ret = SERVICE_CFG_NEW;
    }

    return ret;
}

/*
 * Get InitD service configuration file status.
 *
 * Returns status of the Service file.
 */
static service_cfg_status_t
serviceCfgStatusInitD(const char *serviceName, uid_t uid, gid_t gid) {
    char cfgScript[PATH_MAX] = {0};
    struct stat st = {0};
    service_cfg_status_t ret = SERVICE_CFG_ERROR;

    if (snprintf(cfgScript, sizeof(cfgScript), "/etc/sysconfig/%s", serviceName) < 0) {
        perror("error: serviceCfgStatusInitD, snprintf failed");
        return ret;
    }

    if (stat(cfgScript, &st) == 0) {
        ret = SERVICE_CFG_EXIST;
    } else {
        ret = SERVICE_CFG_NEW;
    }
    return ret;
}

/*
 * Get OpenRc service configuration file status.
 *
 * Returns status of the Service file.
 */
static service_cfg_status_t
serviceCfgStatusOpenRc(const char *serviceName, uid_t uid, gid_t gid) {
    char cfgScript[PATH_MAX] = {0};
    struct stat st = {0};
    service_cfg_status_t ret = SERVICE_CFG_ERROR;

    if (snprintf(cfgScript, sizeof(cfgScript), "/etc/conf.d/%s", serviceName) < 0) {
        perror("error: serviceCfgStatusOpenRc, snprintf failed");
        return ret;
    }

    if (stat(cfgScript, &st) == 0) {
        ret = SERVICE_CFG_EXIST;
    } else {
        ret = SERVICE_CFG_NEW;
    }
    return ret;
}

/*
 * Setup new service configuration for Systemd service.
 *
 * Returns SERVICE_STATUS_SUCCESS if service was setup correctly, other values in case of failure.
 */
static service_status_t
newServiceCfgSystemD(const char *serviceCfgPath, const char *libscopePath, uid_t uid, gid_t gid) {
    service_status_t res = SERVICE_STATUS_SUCCESS;
    char cfgEntry[BUFSIZE] = {0};

    FILE *fPtr = nsFileFopen(serviceCfgPath, "a", uid, gid, geteuid(), getegid());
    if (fPtr == NULL) {
        fprintf(stderr, "\nerror: newServiceCfgSystemD, fopen failed");
        return SERVICE_STATUS_ERROR_OTHER;
    }

    size_t size = snprintf(cfgEntry, BUFSIZE, "[Service]\nEnvironment=LD_PRELOAD=%s\n", libscopePath);
    if (fwrite(cfgEntry, sizeof(char), size, fPtr) < size) {
        perror("error: newServiceCfgSystemD, fwrite failed");
        res = SERVICE_STATUS_ERROR_OTHER;
    }

    fclose(fPtr);

    return res;
}

/*
 * Setup new service configuration for initD service.
 *
 * Returns SERVICE_STATUS_SUCCESS if service was setup correctly, other values in case of failure.
 */
static service_status_t
newServiceCfgInitD(const char *serviceCfgPath, const char *libscopePath, uid_t uid, gid_t gid) {
    service_status_t res = SERVICE_STATUS_SUCCESS;
    char cfgEntry[BUFSIZE] = {0};

    FILE *fPtr = nsFileFopen(serviceCfgPath, "a", uid, gid, geteuid(), getegid());

    if (fPtr == NULL) {
        fprintf(stderr, "\nerror: newServiceCfgInitD, fopen failed");
        return SERVICE_STATUS_ERROR_OTHER;
    }

    size_t size = snprintf(cfgEntry, BUFSIZE, "LD_PRELOAD=%s\n", libscopePath);
    if (fwrite(cfgEntry, sizeof(char), size, fPtr) < size) {
        perror("error: newServiceCfgInitD, fwrite failed");
        res = SERVICE_STATUS_ERROR_OTHER;
    }

    fclose(fPtr);

    return res;
}

/*
 * Setup new service configuration for OpenRc service.
 *
 * Returns SERVICE_STATUS_SUCCESS if service was setup correctly, other values in case of failure.
 */
static service_status_t
newServiceCfgOpenRc(const char *serviceCfgPath, const char *libscopePath, uid_t uid, gid_t gid) {
    service_status_t res = SERVICE_STATUS_SUCCESS;
    char cfgEntry[BUFSIZE] = {0};

    FILE *fPtr = nsFileFopen(serviceCfgPath, "a", uid, gid, geteuid(), getegid());

    if (fPtr == NULL) {
        fprintf(stderr, "\nerror: newServiceCfgOpenRc, fopen failed");
        return SERVICE_STATUS_ERROR_OTHER;
    }

    size_t size = snprintf(cfgEntry, BUFSIZE, "export LD_PRELOAD=%s\n", libscopePath);
    if (fwrite(cfgEntry, sizeof(char), size, fPtr) < size) {
        perror("error: newServiceCfgOpenRc, fwrite failed");
        res = SERVICE_STATUS_ERROR_OTHER;
    }

    fclose(fPtr);

    return res;
}

/*
 * Modify configuration for Systemd service.
 *
 * Returns SERVICE_STATUS_SUCCESS if service was modified correctly, other values in case of failure.
 */
static service_status_t
modifyServiceCfgSystemd(const char *serviceCfgPath, const char *libscopePath, uid_t nsEuid, gid_t nsEgid) {
    FILE *readFd;
    FILE *newFd;
    char *tempPath = "/tmp/tmpFile-XXXXXX";
    bool serviceSectionFound = FALSE;
    char cfgEntry[BUFSIZE] = {0};

    if ((readFd = fopen(serviceCfgPath, "r")) == NULL) {
        perror("error: modifyServiceCfgSystemd, fopen serviceFile failed");
        return SERVICE_STATUS_ERROR_OTHER;
    }

    uid_t eUid = geteuid();
    gid_t eGid = getegid();

    if ((newFd = nsFileFopen(tempPath, "w+", nsEuid, nsEgid, eUid, eGid)) == NULL) {
        fprintf(stderr, "\nerror: modifyServiceCfgSystemd, nsFileFopen tempFile failed");
        fclose(readFd);
        return SERVICE_STATUS_ERROR_OTHER;
    }

    snprintf(cfgEntry, BUFSIZE, "[Service]\nEnvironment=LD_PRELOAD=%s\n", libscopePath);

    while (!feof(readFd)) {
        char buf[4096] = {0};
        int res = fscanf(readFd, "%s", buf);

        if (strcmp(buf, "[Service]") == 0) {
            serviceSectionFound = TRUE;
            fprintf(newFd, "%s", cfgEntry);
        } else if (res == 0){
            fprintf(newFd, "%s ", buf);
        }
    }

    // the file was empty
    if (serviceSectionFound == FALSE) {
        fprintf(newFd, "%s", cfgEntry);
    }

    fclose(newFd);
    fclose(readFd);

    if (nsFileRename(tempPath, serviceCfgPath, nsEuid ,nsEgid, eUid, eGid)) {
        fprintf(stderr, "\nerror: modifyServiceCfgSystemd, nsFileRename failed");
    }
    unlink(tempPath);

    return SERVICE_STATUS_SUCCESS;
}

static service_status_t
removeServiceCfgsSystemd(void) {
    service_status_t res = SERVICE_STATUS_SUCCESS;
    char cfgScript[PATH_MAX] = {0};
    struct stat st = {0};
    DIR *d;
    struct dirent *dir;

    // Remove scope from SystemD service configs
    const char *const systemDPrefixList[] = {
        "/etc/systemd/system",
        "/lib/systemd/system",
        "/run/systemd/system",
        "/usr/lib/systemd/system"
    };
    // For each systemd dir location
    for (int i = 0; i < sizeof(systemDPrefixList)/sizeof(char*); ++i) {
        d = opendir(systemDPrefixList[i]);
        if (d) {
            // For each service directory
            while ((dir = readdir(d)) != NULL) {
                // Look for the presence of an env.conf file
                if (snprintf(cfgScript, sizeof(cfgScript), "%s/%s/env.conf", 
                            systemDPrefixList[i], dir->d_name) < 0) {
                    perror("error: setupUnservice, snprintf failed");
                    res = SERVICE_STATUS_ERROR_OTHER;
                    continue;
                }
                if (stat(cfgScript, &st) == 0) {
                    // If a service is configured with scope, remove scope from it
                    if (isCfgFileConfigured(cfgScript)) {
                        if (removeScopeCfgFile(cfgScript) <= 0) {
                            res = SERVICE_STATUS_ERROR_OTHER;
                        }
                    }
                }
            }
            closedir(d);
        }
    }

    return res;
}

static service_status_t
removeServiceCfgsInitD(void) {
    service_status_t res = SERVICE_STATUS_SUCCESS;
    char cfgScript[PATH_MAX] = {0};
    DIR *d;
    struct dirent *dir;

    // Open the the openrc dir location
    d = opendir("/etc/sysconfig");
    if (d) {
        // For each service file
        while ((dir = readdir(d)) != NULL) {
            if (snprintf(cfgScript, sizeof(cfgScript), "/etc/sysconfig/%s", dir->d_name) < 0) {
                perror("error: removeServiceCfgsInitD, snprintf failed");
                res = SERVICE_STATUS_ERROR_OTHER;
                continue;
            }
            // If a service is configured with scope, remove scope from it
            if (isCfgFileConfigured(cfgScript)) {
                if (removeScopeCfgFile(cfgScript) <= 0) {
                    res = SERVICE_STATUS_ERROR_OTHER;
                }
            }
        }
        closedir(d);
    }
    
    return res;
}

static service_status_t
removeServiceCfgsOpenRC(void) {
    service_status_t res = SERVICE_STATUS_SUCCESS;
    char cfgScript[PATH_MAX] = {0};
    DIR *d;
    struct dirent *dir;

    // Open the the initd dir location
    d = opendir("/etc/init.d");
    if (d) {
        // For each service file
        while ((dir = readdir(d)) != NULL) {
            if (snprintf(cfgScript, sizeof(cfgScript), "/etc/init.d/%s", dir->d_name) < 0) {
                perror("error: removeServiceCfgsOpenRC, snprintf failed");
                res = SERVICE_STATUS_ERROR_OTHER;
                continue;
            }
            // If a service is configured with scope, remove scope from it
            if (isCfgFileConfigured(cfgScript)) {
                if (removeScopeCfgFile(cfgScript) <= 0) {
                    res = SERVICE_STATUS_ERROR_OTHER;
                }
            }
        }
        closedir(d);
    }
    
    return res;
}

// Service manager base class
struct service_ops {
    bool (*isServiceInstalled)(const char *serviceName);
    service_cfg_status_t (*serviceCfgStatus)(const char *serviceCfgPath, uid_t nsUid, gid_t nsGid);
    service_status_t (*newServiceCfg)(const char *serviceCfgPath, const char *libscopePath, uid_t nsUid, gid_t nsGid);
    service_status_t (*modifyServiceCfg)(const char *serviceCfgPath, const char *libscopePath, uid_t nsUid, gid_t nsGid);
    service_status_t (*removeAllScopeServiceCfg)(void);
};

static struct service_ops SystemD = {
    .isServiceInstalled = isServiceInstalledSystemD,
    .serviceCfgStatus = serviceCfgStatusSystemD,
    .newServiceCfg = newServiceCfgSystemD,
    .modifyServiceCfg = modifyServiceCfgSystemd,
    .removeAllScopeServiceCfg = removeServiceCfgsSystemd,
};

static struct service_ops InitD = {
    .isServiceInstalled = isServiceInstalledInitDOpenRc,
    .serviceCfgStatus = serviceCfgStatusInitD,
    .newServiceCfg = newServiceCfgInitD,
    .modifyServiceCfg = newServiceCfgInitD,
    .removeAllScopeServiceCfg = removeServiceCfgsInitD,
};

static struct service_ops OpenRc = {
    .isServiceInstalled = isServiceInstalledInitDOpenRc,
    .serviceCfgStatus = serviceCfgStatusOpenRc,
    .newServiceCfg = newServiceCfgOpenRc,
    .modifyServiceCfg = newServiceCfgOpenRc,
    .removeAllScopeServiceCfg = removeServiceCfgsOpenRC,
};

/*
 * Setup a specific service
 * Returns SERVICE_STATUS_SUCCESS if service was setup correctly, other values in case of failure.
 */
service_status_t
setupService(const char *serviceName, uid_t nsUid, gid_t nsGid) {
    struct stat sb = {0};
    struct service_ops *service;

    char serviceCfgPath[PATH_MAX] = {0};
    char libscopePath[PATH_MAX] = {0};

    service_status_t status;

    if (stat(OPENRC_DIR, &sb) == 0) {
        service = &OpenRc;
        if (snprintf(serviceCfgPath, sizeof(serviceCfgPath), "/etc/conf.d/%s", serviceName) < 0) {
            perror("error: setupService, snprintf OpenRc failed");
            return SERVICE_STATUS_ERROR_OTHER;
        }
    } else if (stat(SYSTEMD_DIR, &sb) == 0) {
        service = &SystemD;
        memset(serviceCfgPath, 0, PATH_MAX);
        if (snprintf(serviceCfgPath, sizeof(serviceCfgPath), "/etc/systemd/system/%s.service.d/env.conf", serviceName) < 0) {
            perror("error: setupService, snprintf SystemD failed");
            return SERVICE_STATUS_ERROR_OTHER;
        }
    } else if (stat(INITD_DIR, &sb) == 0) {
        service = &InitD;
        memset(serviceCfgPath, 0, PATH_MAX);
        if (snprintf(serviceCfgPath, sizeof(serviceCfgPath), "/etc/sysconfig/%s", serviceName) < 0) {
            perror("error: setupService, snprintf InitD failed");
            return SERVICE_STATUS_ERROR_OTHER;
        }
    } else {
        fprintf(stderr, "error: unknown boot system\n");
        return SERVICE_STATUS_ERROR_OTHER;
    }

    if (service->isServiceInstalled(serviceName) == FALSE) {
        fprintf(stderr, "info: service %s is not installed\n", serviceName);
        return SERVICE_STATUS_NOT_INSTALLED;
    }

    const char *loaderVersion = libverNormalizedVersion(SCOPE_VER);
    bool isDevVersion = libverIsNormVersionDev(loaderVersion);

    snprintf(libscopePath, PATH_MAX, "/usr/lib/appscope/%s/libscope.so", loaderVersion);
    if (access(libscopePath, R_OK) || isDevVersion) {
        memset(libscopePath, 0, PATH_MAX);
        snprintf(libscopePath, PATH_MAX, "/tmp/appscope/%s/libscope.so", loaderVersion);
        if (access(libscopePath, R_OK)) {
            fprintf(stderr, "error: libscope is not available %s\n", libscopePath);
            return SERVICE_STATUS_ERROR_OTHER;
        }
    }

    service_cfg_status_t cfgStatus = service->serviceCfgStatus(serviceName, nsUid, nsGid);
    if (cfgStatus == SERVICE_CFG_ERROR) {
        return SERVICE_STATUS_ERROR_OTHER;
    } else if (cfgStatus == SERVICE_CFG_NEW) {
        // Fresh configuration
        status = service->newServiceCfg(serviceCfgPath, libscopePath, nsUid, nsGid);
    } else if (isCfgFileConfigured(serviceCfgPath) == FALSE) {
        // Modification of configuration file
        status = service->modifyServiceCfg(serviceCfgPath, libscopePath, nsUid, nsGid);
    } else {
        // Service was already setup correctly
        return SERVICE_STATUS_SUCCESS;
    }

    // Change permission and ownership if modify or create was success
    if (status == SERVICE_STATUS_SUCCESS ) {
        chmod(serviceCfgPath, 0644);
    }
    
    return status;

}

/*
 * Remove scope from all service configurations
 * Returns SERVICE_STATUS_SUCCESS on success
 */
service_status_t
setupUnservice(void) {
    service_status_t res = SERVICE_STATUS_SUCCESS;
    struct service_ops serviceMgrs[] = {SystemD, InitD, OpenRc};

    for (int i = 0; i < (sizeof(serviceMgrs) / sizeof(struct service_ops)); i++) {
        if ((res = serviceMgrs[i].removeAllScopeServiceCfg()) != SERVICE_STATUS_SUCCESS) {
            return res;
        }
    }

    return res;
}

/*
 * Load File into memory.
 * Returns memory address in case of success, NULL otherwise.
 */
char *
setupLoadFileIntoMem(size_t *size, const char *path)
{
    // Load file into memory
    char *resMem = NULL;
    int fd;

    if (path == NULL) {
        return resMem;
    }

    if ((fd = open(path, O_RDONLY)) == -1) {
        perror("open failed");
        goto closeFd;
    }

    *size = lseek(fd, 0, SEEK_END);
    if (*size == (off_t)-1) {
        perror("lseek failed");
        goto closeFd;
    }

    resMem = mmap(NULL, *size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (resMem == MAP_FAILED) {
        perror("mmap failed");
        resMem = NULL;
        goto closeFd;
    }

closeFd:
    close(fd);

    return resMem;
}

// Mount the scope filter and unix socket into mountDest/usr/lib/appscope/*
int
setupMount(const char *mountDest, uid_t nsUid, gid_t nsGid)
{
    // TODO implement

    return 0;
}

// Install a filter file in /usr/lib/appscope/
int
setupFilter(void *filterFileMem, size_t filterSize, uid_t nsUid, gid_t nsGid)
{
    int filterFd;
    bool status = FALSE;

    if (libdirCreateDirIfMissing(SCOPE_USR_PATH, 0664, nsUid, nsGid) > MKDIR_STATUS_EXISTS) {
        fprintf(stderr, "error: setupFilter: failed to create directory\n");
        return status;
    }

    if ((filterFd = nsFileOpenWithMode(SCOPE_FILTER_USR_PATH, O_RDWR | O_CREAT, 0664, nsUid, nsGid, geteuid(), getegid())) == -1) {
        return status;
    }

    if (ftruncate(filterFd, filterSize) != 0) {
        goto cleanupDestFd;
    }

    char *dest = mmap(NULL, filterSize, PROT_READ | PROT_WRITE, MAP_SHARED, filterFd, 0);
    if (dest == MAP_FAILED) {
        goto cleanupDestFd;
    }

    memcpy(dest, filterFileMem, filterSize);

    status = TRUE;

cleanupDestFd:
    close(filterFd);

    return status;
}

// Set ld.so.preload to point to libscope
int
setupPreload(uid_t nsUid, gid_t nsGid)
{
    char *scopeLibPath;
    char buf[PATH_MAX] = {0};

    scopeLibPath = (char *)libdirGetPath(LIBRARY_FILE);

    if (access(scopeLibPath, R_OK|X_OK)) {
        fprintf(stderr, "error: library %s is missing, not readable, or not executable\n", scopeLibPath);
        return FALSE;
    }

    int fd = nsFileOpenWithMode("/etc/ld.so.preload", O_CREAT | O_RDWR | O_TRUNC, 0644, nsUid, nsGid, geteuid(), getegid());
    if (fd < 0) {
        perror("fopen failed");
        return FALSE;
    }

    size_t len = snprintf(buf, sizeof(buf), scopeLibPath);
    if (write(fd, buf, len) != len) {
        perror("write failed");
        close(fd);
        return FALSE;
    }

    if (close(fd) != 0) {
        perror("close failed");
        return FALSE;
    }

    return TRUE;
}


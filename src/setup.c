#define _GNU_SOURCE
#include <fcntl.h>
#include <stdlib.h>

#include "nsfile.h"
#include "loaderop.h"
#include "libdir.h"
#include "libver.h"
#include "setup.h"
#include "scopestdlib.h"

#define BUFSIZE (4096)

#define OPENRC_DIR "/etc/rc.conf"
#define SYSTEMD_DIR "/etc/systemd"
#define INITD_DIR "/etc/init.d"
#define PROFILE_SCRIPT "#! /bin/bash\nlib_found=0\nfilter_found=0\nif test -f /usr/lib/appscope/%s/libscope.so; then\n    lib_found=1\nfi\nif test -f /usr/lib/appscope/scope_filter; then\n    filter_found=1\nelif test -f /tmp/appscope/scope_filter; then\n    filter_found=1\nfi\nif [ $lib_found == 1 ] && [ $filter_found == 1 ]; then\n    export LD_PRELOAD=\"%s $LD_PRELOAD\"\nfi\n"

typedef enum {
    SERVICE_CFG_ERROR,
    SERVICE_CFG_NEW,
    SERVICE_CFG_EXIST,
} service_cfg_status_t;

struct service_ops {
    bool (*isServiceInstalled)(const char *serviceName);
    service_cfg_status_t (*serviceCfgStatus)(const char *serviceCfgPath, uid_t nsUid, gid_t nsGid);
    service_status_t (*newServiceCfg)(const char *serviceCfgPath, const char *libscopePath, uid_t nsUid, gid_t nsGid);
    service_status_t (*modifyServiceCfg)(const char *serviceCfgPath, const char *libscopePath, uid_t nsUid, gid_t nsGid);
};

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
        if (scope_snprintf(cfgPath, sizeof(cfgPath), "%s/%s.service", servicePrefixList[i], serviceName) < 0) {
            scope_perror("error: isServiceInstalledSystemD, scope_snprintf failed");
            return FALSE;
        }

        if (scope_stat(cfgPath, &st) == 0) {
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
    if (scope_snprintf(cfgPath, sizeof(cfgPath), "/etc/init.d/%s", serviceName) < 0) {
        scope_perror("error: isServiceInstalledInitDOpenRc, scope_snprintf failed");
        return FALSE;
    }

    return (scope_stat(cfgPath, &st) == 0) ? TRUE : FALSE;
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

    if (scope_snprintf(cfgScript, sizeof(cfgScript), "/etc/systemd/system/%s.service.d/", serviceName) < 0) {
        scope_perror("error: serviceCfgStatusSystemD, scope_snprintf failed");
        return ret;
    }

    // create service.d directory if it does not exists.
    if (scope_stat(cfgScript, &st) != 0) {
        if (nsFileMkdir(cfgScript, 0755, uid, gid, scope_geteuid(), scope_getegid()) != 0) {
            scope_perror("error: serviceCfgStatusSystemD, scope_mkdir failed");
            return ret;
        }
    }

    scope_strncat(cfgScript, "env.conf", sizeof("env.conf") - 1);

    if (scope_stat(cfgScript, &st) == 0) {
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

    if (scope_snprintf(cfgScript, sizeof(cfgScript), "/etc/sysconfig/%s", serviceName) < 0) {
        scope_perror("error: serviceCfgStatusInitD, scope_snprintf failed");
        return ret;
    }

    if (scope_stat(cfgScript, &st) == 0) {
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

    if (scope_snprintf(cfgScript, sizeof(cfgScript), "/etc/conf.d/%s", serviceName) < 0) {
        scope_perror("error: serviceCfgStatusOpenRc, scope_snprintf failed");
        return ret;
    }

    if (scope_stat(cfgScript, &st) == 0) {
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

    FILE *fPtr = nsFileFopen(serviceCfgPath, "a", uid, gid, scope_geteuid(), scope_getegid());
    if (fPtr == NULL) {
        scope_fprintf(scope_stderr, "\nerror: newServiceCfgSystemD, scope_fopen failed");
        return SERVICE_STATUS_ERROR_OTHER;
    }

    size_t size = scope_snprintf(cfgEntry, BUFSIZE, "[Service]\nEnvironment=LD_PRELOAD=%s\n", libscopePath);
    if (scope_fwrite(cfgEntry, sizeof(char), size, fPtr) < size) {
        scope_perror("error: newServiceCfgSystemD, scope_fwrite failed");
        res = SERVICE_STATUS_ERROR_OTHER;
    }

    scope_fclose(fPtr);

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

    FILE *fPtr = nsFileFopen(serviceCfgPath, "a", uid, gid, scope_geteuid(), scope_getegid());

    if (fPtr == NULL) {
        scope_fprintf(scope_stderr, "\nerror: newServiceCfgInitD, scope_fopen failed");
        return SERVICE_STATUS_ERROR_OTHER;
    }

    size_t size = scope_snprintf(cfgEntry, BUFSIZE, "LD_PRELOAD=%s\n", libscopePath);
    if (scope_fwrite(cfgEntry, sizeof(char), size, fPtr) < size) {
        scope_perror("error: newServiceCfgInitD, scope_fwrite failed");
        res = SERVICE_STATUS_ERROR_OTHER;
    }

    scope_fclose(fPtr);

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

    FILE *fPtr = nsFileFopen(serviceCfgPath, "a", uid, gid, scope_geteuid(), scope_getegid());

    if (fPtr == NULL) {
        scope_fprintf(scope_stderr, "\nerror: newServiceCfgOpenRc, scope_fopen failed");
        return SERVICE_STATUS_ERROR_OTHER;
    }

    size_t size = scope_snprintf(cfgEntry, BUFSIZE, "export LD_PRELOAD=%s\n", libscopePath);
    if (scope_fwrite(cfgEntry, sizeof(char), size, fPtr) < size) {
        scope_perror("error: newServiceCfgOpenRc, scope_fwrite failed");
        res = SERVICE_STATUS_ERROR_OTHER;
    }

    scope_fclose(fPtr);

    return res;
}

/*
 * Check if specified service needs to be configured.
 *
 * Returns TRUE if service is already configured FALSE otherwise.
 */
static bool
isCfgFileConfigured(const char *serviceCfgPath) {
    FILE *fPtr;
    int res = FALSE;
    char buf[BUFSIZE] = {0};

    if ((fPtr = scope_fopen(serviceCfgPath, "r")) == NULL) {
        scope_perror("isCfgFileConfigured scope_fopen failed");
        return res;
    }

    while(scope_fgets(buf, sizeof(buf), fPtr)) {
        // TODO improve it to verify particular version ?
        if (scope_strstr(buf, "/libscope.so")) {
            res = TRUE;
            break;
        }
    }

    scope_fclose(fPtr);

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

    if ((readFd = scope_fopen(serviceCfgPath, "r")) == NULL) {
        scope_perror("error: modifyServiceCfgSystemd, scope_fopen serviceFile failed");
        return SERVICE_STATUS_ERROR_OTHER;
    }

    uid_t eUid = scope_geteuid();
    gid_t eGid = scope_getegid();

    if ((newFd = nsFileFopen(tempPath, "w+", nsEuid, nsEgid, eUid, eGid)) == NULL) {
        scope_fprintf(scope_stderr, "\nerror: modifyServiceCfgSystemd, nsFileFopen tempFile failed");
        scope_fclose(readFd);
        return SERVICE_STATUS_ERROR_OTHER;
    }

    scope_snprintf(cfgEntry, BUFSIZE, "[Service]\nEnvironment=LD_PRELOAD=%s\n", libscopePath);

    while (!scope_feof(readFd)) {
        char buf[4096] = {0};
        int res = scope_fscanf(readFd, "%s", buf);

        if (scope_strcmp(buf, "[Service]") == 0) {
            serviceSectionFound = TRUE;
            scope_fprintf(newFd, "%s", cfgEntry);
        } else if (res == 0){
            scope_fprintf(newFd, "%s ", buf);
        }
    }

    // the file was empty
    if (serviceSectionFound == FALSE) {
        scope_fprintf(newFd, "%s", cfgEntry);
    }

    scope_fclose(newFd);
    scope_fclose(readFd);

    if (nsFileRename(tempPath, serviceCfgPath, nsEuid ,nsEgid, eUid, eGid)) {
        scope_fprintf(scope_stderr, "\nerror: modifyServiceCfgSystemd, nsFileRename failed");
    }
    scope_unlink(tempPath);

    return SERVICE_STATUS_SUCCESS;
}

static struct service_ops SystemDService = {
    .isServiceInstalled = isServiceInstalledSystemD,
    .serviceCfgStatus = serviceCfgStatusSystemD,
    .newServiceCfg = newServiceCfgSystemD,
    .modifyServiceCfg= modifyServiceCfgSystemd,
};

static struct service_ops InitDService = {
    .isServiceInstalled = isServiceInstalledInitDOpenRc,
    .serviceCfgStatus = serviceCfgStatusInitD,
    .newServiceCfg = newServiceCfgInitD,
    .modifyServiceCfg= newServiceCfgInitD,
};

static struct service_ops OpenRcService = {
    .isServiceInstalled = isServiceInstalledInitDOpenRc,
    .serviceCfgStatus = serviceCfgStatusOpenRc,
    .newServiceCfg = newServiceCfgOpenRc,
    .modifyServiceCfg= newServiceCfgOpenRc,
};

/*
 * Setup specific service.
 *
 * Returns SERVICE_STATUS_SUCCESS if service was setup correctly, other values in case of failure.
 */
service_status_t
setupService(const char *serviceName, uid_t nsUid, gid_t nsGid) {
    struct stat sb = {0};
    struct service_ops *service;

    char serviceCfgPath[PATH_MAX] = {0};
    char libscopePath[PATH_MAX] = {0};

    service_status_t status;

    if (scope_stat(OPENRC_DIR, &sb) == 0) {
        service = &OpenRcService;
        if (scope_snprintf(serviceCfgPath, sizeof(serviceCfgPath), "/etc/conf.d/%s", serviceName) < 0) {
            scope_perror("error: setupService, scope_snprintf OpenRc failed");
            return SERVICE_STATUS_ERROR_OTHER;
        }
    } else if (scope_stat(SYSTEMD_DIR, &sb) == 0) {
        service = &SystemDService;
        scope_memset(serviceCfgPath, 0, PATH_MAX);
        if (scope_snprintf(serviceCfgPath, sizeof(serviceCfgPath), "/etc/systemd/system/%s.service.d/env.conf", serviceName) < 0) {
            scope_perror("error: setupService, scope_snprintf SystemD failed");
            return SERVICE_STATUS_ERROR_OTHER;
        }
    } else if (scope_stat(INITD_DIR, &sb) == 0) {
        service = &InitDService;
        scope_memset(serviceCfgPath, 0, PATH_MAX);
        if (scope_snprintf(serviceCfgPath, sizeof(serviceCfgPath), "/etc/sysconfig/%s", serviceName) < 0) {
            scope_perror("error: setupService, scope_snprintf InitD failed");
            return SERVICE_STATUS_ERROR_OTHER;
        }
    } else {
        scope_fprintf(scope_stderr, "error: unknown boot system\n");
        return SERVICE_STATUS_ERROR_OTHER;
    }

    if (service->isServiceInstalled(serviceName) == FALSE) {
        scope_fprintf(scope_stderr, "info: service %s is not installed\n", serviceName);
        return SERVICE_STATUS_NOT_INSTALLED;
    }

    const char *loaderVersion = libverNormalizedVersion(SCOPE_VER);
    bool isDevVersion = libverIsNormVersionDev(loaderVersion);

    scope_snprintf(libscopePath, PATH_MAX, "/usr/lib/appscope/%s/libscope.so", loaderVersion);
    if (scope_access(libscopePath, R_OK) || isDevVersion) {
        scope_memset(libscopePath, 0, PATH_MAX);
        scope_snprintf(libscopePath, PATH_MAX, "/tmp/appscope/%s/libscope.so", loaderVersion);
        if (scope_access(libscopePath, R_OK)) {
            scope_fprintf(scope_stderr, "error: libscope is not available %s\n", libscopePath);
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
        scope_chmod(serviceCfgPath, 0644);
    }
    
    return status;
}


 /*
 * Setup the /etc/profile scope startup script
 * Returns status of operation TRUE in case of success, FALSE otherwise
 */
static bool
setupProfile(const char *libscopePath, const char *loaderVersion, uid_t nsUid, gid_t nsGid) {

    /*
     * If the env var is set to anything, return.
     * Should use the utils func for this. However, it
     * requires libscope specifics that we don't want here.
     * Should fix that.
     */
    if (getenv("SCOPE_START_NOPROFILE")) {
        return TRUE;
    }

    char buf[PATH_MAX] = {0};
    int fd = nsFileOpenWithMode("/etc/profile.d/scope.sh", O_CREAT | O_RDWR | O_TRUNC, 0644, nsUid, nsGid, scope_geteuid(), scope_getegid());
    if (fd < 0) {
        return FALSE;
    }

    size_t len = scope_snprintf(buf, sizeof(buf), PROFILE_SCRIPT, loaderVersion, libscopePath);
    if (scope_write(fd, buf, len) != len) {
        scope_perror("scope_write failed");
        scope_close(fd);
        return FALSE;
    }

    if (scope_close(fd) != 0) {
        scope_perror("scope_fopen failed");
        return FALSE;
    }

    return TRUE;
}

 /*
 * Extract memory to specific filter path file
 *
 * Returns status of operation TRUE in case of success, FALSE otherwise
 */
static bool
setupExtractFilterFile(void *filterFileMem, size_t filterSize, const char *outputFilterPath, uid_t nsUid, gid_t nsGid) {
    int filterFd;
    bool status = FALSE;

    if ((filterFd = nsFileOpenWithMode(outputFilterPath, O_RDWR | O_CREAT, 0664, nsUid, nsGid, scope_geteuid(), scope_getegid())) == -1) {
        return status;
    }

    if (scope_ftruncate(filterFd, filterSize) != 0) {
        goto cleanupDestFd;
    }

    char *dest = scope_mmap(NULL, filterSize, PROT_READ | PROT_WRITE, MAP_SHARED, filterFd, 0);
    if (dest == MAP_FAILED) {
        goto cleanupDestFd;
    }

    scope_memcpy(dest, filterFileMem, filterSize);

    status = TRUE;

cleanupDestFd:

    scope_close(filterFd);

    return status;
}

/*
 * Load File into memory.
 *
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

    if ((fd = scope_open(path, O_RDONLY)) == -1) {
        scope_perror("scope_open failed");
        goto closeFd;
    }

    *size = scope_lseek(fd, 0, SEEK_END);
    if (*size == (off_t)-1) {
        scope_perror("scope_lseek failed");
        goto closeFd;
    }

    resMem = scope_mmap(NULL, *size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (resMem == MAP_FAILED) {
        scope_perror("scope_mmap failed");
        resMem = NULL;
        goto closeFd;
    }

closeFd:

    scope_close(fd);

    return resMem;
}

 /*
 * Configure the environment
 * - setup /etc/profile.d/scope.sh
 * - extract memory to filter file /usr/lib/appscope/scope_filter or /tmp/appscope/scope_filter
 * - extract libscope.so to /usr/lib/appscope/<version>/libscope.so or /tmp/appscope/<version>/libscope.so if it doesn't exists
 * - patch the library
 * Returns status of operation 0 in case of success, other value otherwise
 */
int
setupConfigure(void *filterFileMem, size_t filterSize, uid_t nsUid, gid_t nsGid) {
    char path[PATH_MAX] = {0};
    mode_t mode = 0755;

    // Create destination directory if not exists
    const char *loaderVersion = libverNormalizedVersion(SCOPE_VER);
    bool isDevVersion = libverIsNormVersionDev(loaderVersion);
    bool overwrite = isDevVersion;

    /*
     * A profile will not be configured with a dev version value.
     * Force a profile update if the env var is present.
     */
    if (getenv("SCOPE_START_FORCE_PROFILE")) isDevVersion = FALSE;

    scope_snprintf(path, PATH_MAX, "/usr/lib/appscope/%s/", loaderVersion);
    mkdir_status_t res = libdirCreateDirIfMissing(path, mode, nsUid, nsGid);
    if ((res > MKDIR_STATUS_EXISTS) || isDevVersion) {
        mode = 0777;
        scope_memset(path, 0, PATH_MAX);
        scope_snprintf(path, PATH_MAX, "/tmp/appscope/%s/", loaderVersion);
        mkdir_status_t res = libdirCreateDirIfMissing(path, mode, nsUid, nsGid);
        if (res > MKDIR_STATUS_EXISTS) {
            scope_fprintf(scope_stderr, "setupConfigure: libdirCreateDirIfMissing failed\n");
            return -1;
        }
    }

    scope_strncat(path, "libscope.so", sizeof("libscope.so"));

    // Extract[create] the filter file to filter location
    if (setupExtractFilterFile(filterFileMem, filterSize, SCOPE_FILTER_USR_PATH, nsUid, nsGid) == FALSE) {
        if (setupExtractFilterFile(filterFileMem, filterSize, SCOPE_FILTER_TMP_PATH, nsUid, nsGid) == FALSE) {
            scope_fprintf(scope_stderr, "setupConfigure: setup filter file failed\n");
            return -1;
        }
    }

    /*
     * Setup /etc/profile.d/scope.sh
     * Only update the profile if we are using the system dir.
     */
    if (scope_strstr(path, "/usr/lib")) {
        if (setupProfile(path, loaderVersion, nsUid, nsGid) == FALSE) {
            scope_fprintf(scope_stderr, "setupConfigure: setupProfile failed\n");
            return -1;
        }
    }

    // Extract libscope.so
    if (libdirSaveLibraryFile(path, overwrite, mode, nsUid, nsGid)) {
        scope_fprintf(scope_stderr, "setupConfigure: saving %s failed\n", path);
        return -1;
    }

    // Patch the library
    if (loaderOpPatchLibrary(path) == PATCH_FAILED) {
        scope_fprintf(scope_stderr, "setupConfigure: patch %s failed\n, path");
        return -1;
    }

    return 0;
}
